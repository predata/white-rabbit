import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# date + (0 * BDay()) makes date the next business day if necessary
# this is a no-op if date is already a business day

def _build_asset_series(asset_series):
    # legacy support for DataFrame
    if isinstance(asset_series, pd.DataFrame):
        asset_series = asset_series.iloc[0]

    ret_series = asset_series.copy()

    cal_days_index = pd.date_range(ret_series.index.min(), ret_series.index.max())
    ret_series = ret_series.reindex(cal_days_index)
    ret_series = ret_series.fillna(method="bfill")
    ret_series.index = ret_series.index.tz_localize(None)
    return ret_series


def _get_signal_exceedance_dates(signal_series, alpha, rolling_window, holding_period, ignore_overlapping):
    # legacy support for DataFrame
    if isinstance(signal_series, pd.DataFrame):
        signal_series = signal_series.iloc[0]

    signal_series.index = pd.to_datetime(signal_series.index)
    signal_series_rolling = signal_series.rolling(rolling_window)
    z_scores = (signal_series - signal_series_rolling.mean()) / signal_series_rolling.std()
    exceedances = z_scores.loc[z_scores > alpha]
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


def _get_trade_returns(asset_series, holding_period, trade_dates, is_buy):
    open_price_series = asset_series.loc[trade_dates]
    close_price_series = asset_series.shift(-holding_period).loc[trade_dates]
    if is_buy:
        trade_returns_series = (close_price_series - open_price_series) / open_price_series
    else:
        trade_returns_series = (open_price_series - close_price_series) / open_price_series

    open_price_series.name = "open_price"
    close_price_series.name = "close_price"
    trade_returns_series.name = "return"

    return {
        "trade_returns": trade_returns_series,
        "open_price": open_price_series,
        "close_price": close_price_series
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


def get_trade_statistics(signal_series, asset_series, alpha, rolling_window, holding_period,
                         ignore_overlapping, is_buy):
    """
    Calculate number of trades, hit rate, and mean return of trade returns

    signal_series: Series of a signal's values with a DatetimeIndex
    asset_series: Series of an asset's prices with a DatetimeIndex
    alpha: Float, number of standard deviations for Bollinger bands
    rolling_window: Integer, days of moving average lookback for Bollinger bands
    holding_period: Integer, days to hold position
    ignore_overlapping: Boolean, ignore overlapping trade triggers
    is_buy: Boolean, action is buy (True) or sell (False)
    """
    exceedances = _get_signal_exceedance_dates(signal_series, alpha, rolling_window,
                                               holding_period, ignore_overlapping)
    trade_dates = exceedances["open_dates"]
    mod_asset_series = _build_asset_series(asset_series)

    trade_returns = _get_trade_returns(mod_asset_series, holding_period, trade_dates, is_buy)
    trade_returns_series = trade_returns["trade_returns"]
    open_price_series = trade_returns["open_price"]
    close_price_series = trade_returns["close_price"]

    # for use with groupby object aggregation
    def _get_positive_trade_pct(returns):
        try:
            return 1.0 * (returns > 0).sum() / returns.count()
        except ZeroDivisionError:
            return np.nan

    if not trade_returns_series.empty:
        trade_returns_groupby = trade_returns_series.groupby(trade_returns_series.index.year)
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
    for year in mod_asset_series.index.year.unique():
        values = summary_statistics_dict.get(year,
                                             {"number_of_trades": 0, "hit_rate": np.nan, "mean_return": np.nan})
        values["year"] = year
        summary_statistics.append(values)
    total_values = {
        "year": "Total",
        "number_of_trades": trade_returns_series.count(),
        "hit_rate": _get_positive_trade_pct(trade_returns_series),
        "mean_return": trade_returns_series.mean(),
    }
    summary_statistics.append(total_values)

    trade_details = pd.concat([trade_returns_series, open_price_series, close_price_series], axis=1)
    trade_details["date"] = trade_details.index.date
    trade_details["close_date"] = exceedances["close_dates"].date
    trade_details["trigger_date"] = exceedances["trigger_dates"].date
    trade_details["z_scores"] = exceedances["z_scores"].values
    trade_details = trade_details.to_dict("records")

    return {
        "summary_statistics": summary_statistics,
        "trade_details": trade_details,
    }


def get_trade_percentiles(signal_series, asset_series, alpha, rolling_window, holding_periods,
                          ignore_overlapping, is_buy, percentiles):
    """
    Calculate percentiles of trade returns

    signal_series: Series of a signal's values with a DatetimeIndex
    asset_series: Series of an asset's prices with a DatetimeIndex
    alpha: Float, number of standard deviations for Bollinger bands
    rolling_window: Integer, days of moving average lookback for Bollinger bands
    holding_periods: List of holding periods
    ignore_overlapping: Boolean, ignore overlapping trade triggers
    is_buy: Boolean, action is buy (True) or sell (False)
    percentiles: List of percentiles to return
    """
    dist = {}
    for holding_period in holding_periods:
        exceedances = _get_signal_exceedance_dates(signal_series, alpha, rolling_window,
                                                   holding_period, ignore_overlapping)
        trade_dates = exceedances["open_dates"]
        mod_asset_series = _build_asset_series(asset_series)

        trade_returns = _get_trade_returns(mod_asset_series, holding_period, trade_dates, is_buy)
        quantiles = [pct / 100.0 for pct in percentiles]
        quantile_values = trade_returns["trade_returns"].quantile(quantiles)
        quantile_values.index = (quantile_values.index * 100).astype("int")
        dist[holding_period] = quantile_values.to_dict()
    return dist


def get_trade_returns(signal_series, asset_series, alpha, rolling_window, holding_period,
                           ignore_overlapping, is_buy):
    """
    Calculate cumulative algo returns, daily algo returns, and cumulative benchmark returns

    signal_series: Series of a signal's values with a DatetimeIndex
    asset_series: Series of an asset's prices with a DatetimeIndex
    alpha: Float, number of standard deviations for Bollinger bands
    rolling_window: Integer, days of moving average lookback for Bollinger bands
    holding_periods: List of holding periods
    ignore_overlapping: Boolean, ignore overlapping trade triggers
    is_buy: Boolean, action is buy (True) or sell (False)
    """
    exceedances = _get_signal_exceedance_dates(signal_series, alpha, rolling_window, holding_period,
                                               ignore_overlapping)
    mod_asset_series = _build_asset_series(asset_series)

    open_trade = (exceedances["open_dates"].value_counts() / mod_asset_series).dropna()
    close_trade = open_trade.copy()
    close_trade.index = close_trade.index + pd.DateOffset(days=holding_period) + (0 * BDay())
    close_trade = close_trade.groupby(close_trade.index).sum().reindex(mod_asset_series.index, fill_value=0)
    open_trade = open_trade.reindex(mod_asset_series.index, fill_value=0)

    cumulative_returns = _get_cumulative_returns(open_trade, close_trade, mod_asset_series, is_buy)
    daily_returns = _get_daily_returns(cumulative_returns)

    first_date = mod_asset_series.index.min()
    last_date = mod_asset_series.index.max()
    start_price = mod_asset_series.loc[first_date]
    benchmark_open_trade = pd.Series(1 / start_price, [first_date])
    benchmark_close_trade = pd.Series(benchmark_open_trade.values, [last_date]).reindex(mod_asset_series.index, fill_value=0)
    benchmark_open_trade = benchmark_open_trade.reindex(mod_asset_series.index, fill_value=0)
    benchmark_returns = _get_cumulative_returns(benchmark_open_trade, benchmark_close_trade, mod_asset_series, is_buy)

    return {
        "cumulative_returns": cumulative_returns,
        "daily_returns": daily_returns,
        "benchmark_returns": benchmark_returns,
    }
