import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# date + (0 * BDay()) makes date the next business day if necessary
# this is a no-op if date is already a business day

def _build_asset_df(asset_df):
    ret_df = asset_df.copy()

    cal_days_index = pd.date_range(ret_df.index.min(), ret_df.index.max())
    ret_df = ret_df.reindex(cal_days_index)
    ret_df = ret_df.fillna(method="bfill")
    ret_df.index = ret_df.index.tz_localize(None)
    return ret_df


def _get_signal_exceedance_dates(signal_df, value_accessor, alpha, rolling_window, holding_period, ignore_overlapping):
    signal_df.index = pd.to_datetime(signal_df.index)
    signal_df_rolling = signal_df.rolling(rolling_window)
    z_scores = (signal_df - signal_df_rolling.mean()) / signal_df_rolling.std()
    exceedances = z_scores.loc[z_scores[value_accessor] > alpha]
    exceedances.columns = ["z_score"]
    exceedance_dates = exceedances.index
    trade_dates = exceedance_dates + BDay()
    close_dates = trade_dates + pd.DateOffset(days=holding_period) + (0 * BDay())

    if ignore_overlapping:
        open_trade = pd.Series(True, trade_dates)
        close_dates_lookup = pd.Series(close_dates, trade_dates)
        close_dates_lookup = close_dates_lookup[~close_dates_lookup.index.duplicated()]
        trade_dates_lookup = pd.Series(trade_dates, exceedance_dates)
        for date in trade_dates:
            if not open_trade[~open_trade.index.duplicated()].loc[date]:
                continue
            open_trade.loc[date + pd.DateOffset(1):close_dates_lookup.loc[date]] = False
        trade_dates = open_trade[open_trade].index
        close_dates = trade_dates + pd.DateOffset(days=holding_period) + (0 * BDay())
        exceedance_dates = trade_dates_lookup[trade_dates_lookup.isin(trade_dates)].index
        exceedances = exceedances.loc[exceedance_dates]

    return {
        "open_dates": trade_dates,
        "close_dates": close_dates,
        "trigger_dates": exceedance_dates,
        "z_scores": exceedances
    }


def _get_trade_returns(asset_df, holding_period, trade_dates, is_buy):
    open_price_df = asset_df.loc[trade_dates]
    close_price_df = asset_df.shift(-holding_period).loc[trade_dates]
    if is_buy:
        trade_returns_df = (close_price_df - open_price_df) / open_price_df
    else:
        trade_returns_df = (open_price_df - close_price_df) / open_price_df

    open_price_df = open_price_df.rename(columns={"price": "open_price"})
    close_price_df = close_price_df.rename(columns={"price": "close_price"})
    trade_returns_df = trade_returns_df.rename(columns={"price": "return"})

    return {
        "trade_returns": trade_returns_df,
        "open_price": open_price_df,
        "close_price": close_price_df
    }


def _get_cumulative_returns(open_pos, close_pos, prices, is_buy):
    # value of open positions + value of closed positions
    if is_buy:
        return ((open_pos - close_pos).cumsum() * prices + (prices * (close_pos - open_pos)).cumsum())
    else:
        return ((close_pos - open_pos).cumsum() * prices + (prices * (open_pos - close_pos)).cumsum())

def _get_daily_returns(cumulative_returns):
    daily_returns = cumulative_returns.apply(np.log1p).diff().apply(np.exp) - 1
    daily_returns[0] = 0
    return daily_returns


def get_trade_statistics(signal_df, value_accessor, asset_df, alpha, rolling_window, holding_period,
                         ignore_overlapping, is_buy):
    """
    Calculate number of trades, hit rate, and mean return of trade returns

    signal_df: DataFrame of a signal's values with a DatetimeIndex and 1 column of values
    value_accessor: Column which contains the data for the provided signal_df
    asset_df: DataFrame with a DatetimeIndex and 1 column named "price"
    alpha: Float, number of standard deviations for Bollinger bands
    rolling_window: Integer, days of moving average lookback for Bollinger bands
    holding_period: Integer, days to hold position
    ignore_overlapping: Boolean, ignore overlapping trade triggers
    is_buy: Boolean, action is buy (True) or sell (False)
    """
    exceedances = _get_signal_exceedance_dates(signal_df, value_accessor, alpha, rolling_window,
                                               holding_period, ignore_overlapping)
    trade_dates = exceedances["open_dates"]
    mod_asset_df = _build_asset_df(asset_df)

    trade_returns = _get_trade_returns(mod_asset_df, holding_period, trade_dates, is_buy)
    trade_returns_df = trade_returns["trade_returns"]
    open_price_df = trade_returns["open_price"]
    close_price_df = trade_returns["close_price"]

    # for use with groupby object aggregation
    def _get_positive_trade_pct(returns):
        try:
            return 1.0 * (returns > 0).sum() / returns.count()
        except ZeroDivisionError:
            return np.nan

    if not trade_returns_df.empty:
        trade_returns_groupby = trade_returns_df.groupby(trade_returns_df.index.year)["return"]
        summary_statistics_columns = {
            "count": "number_of_trades",
            "_get_positive_trade_pct": "hit_rate",
            "mean": "mean_return",
        }
        summary_statistics_dict = trade_returns_groupby.agg(["count", _get_positive_trade_pct, "mean"])\
                                                       .rename(columns=summary_statistics_columns)\
                                                       .to_dict(orient="index")
    else:
        summary_statistics_dict = {}

    summary_statistics = []
    for year in mod_asset_df.index.year.unique():
        values = summary_statistics_dict.get(year,
                                             {"number_of_trades": 0, "hit_rate": np.nan, "mean_return": np.nan})
        values["year"] = year
        summary_statistics.append(values)
    total_values = {
        "year": "Total",
        "number_of_trades": trade_returns_df.count()[0],
        "hit_rate": _get_positive_trade_pct(trade_returns_df)[0],
        "mean_return": trade_returns_df.mean()[0],
    }
    summary_statistics.append(total_values)

    trade_details = pd.concat([trade_returns_df, open_price_df, close_price_df], axis=1)
    trade_details["date"] = trade_details.index.date
    trade_details["close_date"] = exceedances["close_dates"].date
    trade_details["trigger_date"] = exceedances["trigger_dates"].date
    trade_details["z_scores"] = exceedances["z_scores"]["z_score"].values
    trade_details = trade_details.to_dict("records")

    return {
        "summary_statistics": summary_statistics,
        "trade_details": trade_details,
    }


def get_trade_percentiles(signal_df, value_accessor, asset_df, alpha, rolling_window, holding_periods,
                          ignore_overlapping, is_buy, percentiles):
    """
    Calculate percentiles of trade returns

    signal_df: DataFrame of a signal's values with a DatetimeIndex and 1 column of values
    value_accessor: Column which contains the data for the provided signal_df
    asset_df: DataFrame with a DatetimeIndex and 1 column named "price"
    alpha: Float, number of standard deviations for Bollinger bands
    rolling_window: Integer, days of moving average lookback for Bollinger bands
    holding_periods: List of holding periods
    ignore_overlapping: Boolean, ignore overlapping trade triggers
    is_buy: Boolean, action is buy (True) or sell (False)
    percentiles: List of percentiles to return
    """
    dist = {}
    for holding_period in holding_periods:
        exceedances = _get_signal_exceedance_dates(signal_df, value_accessor, alpha, rolling_window,
                                                   holding_period, ignore_overlapping)
        trade_dates = exceedances["open_dates"]
        mod_asset_df = _build_asset_df(asset_df)

        trade_returns = _get_trade_returns(mod_asset_df, holding_period, trade_dates, is_buy)
        quantiles = [pct / 100.0 for pct in percentiles]
        quantile_values = trade_returns["trade_returns"]["return"].quantile(quantiles)
        quantile_values.index = (quantile_values.index * 100).astype("int")
        dist[holding_period] = quantile_values.to_dict()
    return dist


def get_trade_returns(signal_df, value_accessor, asset_df, alpha, rolling_window, holding_period,
                           ignore_overlapping, is_buy):
    """
    Calculate cumulative algo returns, daily algo returns, and cumulative benchmark returns

    signal_df: DataFrame of a signal's values with a DatetimeIndex and 1 column of values
    value_accessor: Column which contains the data for the provided signal_df
    asset_df: DataFrame with a DatetimeIndex and 1 column named "price"
    alpha: Float, number of standard deviations for Bollinger bands
    rolling_window: Integer, days of moving average lookback for Bollinger bands
    holding_periods: List of holding periods
    ignore_overlapping: Boolean, ignore overlapping trade triggers
    is_buy: Boolean, action is buy (True) or sell (False)
    """
    exceedances = _get_signal_exceedance_dates(signal_df, value_accessor, alpha, rolling_window, holding_period,
                                               ignore_overlapping)
    mod_asset_df = _build_asset_df(asset_df)

    open_trade = (exceedances["open_dates"].value_counts() / mod_asset_df["price"]).dropna()
    close_trade = open_trade.copy()
    close_trade.index = close_trade.index + pd.DateOffset(days=holding_period) + (0 * BDay())
    close_trade = close_trade.groupby(close_trade.index).sum().reindex(mod_asset_df.index, fill_value=0)
    open_trade = open_trade.reindex(mod_asset_df.index, fill_value=0)

    cumulative_returns = _get_cumulative_returns(open_trade, close_trade, mod_asset_df["price"], is_buy)
    daily_returns = _get_daily_returns(cumulative_returns)

    first_date = mod_asset_df.index.min()
    last_date = mod_asset_df.index.max()
    start_price = mod_asset_df["price"].loc[first_date]
    benchmark_open_trade = pd.Series(1 / start_price, [first_date])
    benchmark_close_trade = pd.Series(benchmark_open_trade.values, [last_date]).reindex(mod_asset_df.index, fill_value=0)
    benchmark_open_trade = benchmark_open_trade.reindex(mod_asset_df.index, fill_value=0)
    benchmark_returns = _get_cumulative_returns(benchmark_open_trade, benchmark_close_trade, mod_asset_df["price"], is_buy)

    return {
        "cumulative_returns": cumulative_returns,
        "daily_returns": daily_returns,
        "benchmark_returns": benchmark_returns,
    }
