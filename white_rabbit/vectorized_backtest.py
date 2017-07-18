import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


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
    exceedance_dates = exceedances.index
    if ignore_overlapping:
        exceedance_dates = exceedance_dates[~(exceedance_dates.to_series().diff() < pd.Timedelta(days=holding_period))]
    trade_dates = exceedance_dates + BDay()
    close_dates = trade_dates + pd.DateOffset(days=holding_period) + (0 * BDay())

    return {
        "open_dates": trade_dates,
        "close_dates": close_dates
    }


def _get_trade_returns(asset_df, holding_period, trade_dates):
    open_price_df = asset_df.loc[trade_dates]
    close_price_df = asset_df.shift(-holding_period).loc[trade_dates]
    trade_returns_df = (close_price_df - open_price_df) / open_price_df

    open_price_df = open_price_df.rename(columns={"price": "open_price"})
    close_price_df = close_price_df.rename(columns={"price": "close_price"})
    trade_returns_df = trade_returns_df.rename(columns={"price": "return"})

    return {
        "trade_returns": trade_returns_df,
        "open_price": open_price_df,
        "close_price": close_price_df
    }


def get_trade_statistics(signal_df, value_accessor, asset_df, alpha, rolling_window,
                         holding_period, ignore_overlapping):
    """
    signal_df: DataFrame of a signal's values with a DatetimeIndex and 1 column of values
    value_accessor: Column which contains the data for the provided signal_df
    asset_df: DataFrame with a DatetimeIndex and 1 column named "price"
    holding_period: Either a single holding period, or an iterable with multiple periods
    """
    exceedances = _get_signal_exceedance_dates(signal_df, value_accessor, alpha, rolling_window,
                                               holding_period, ignore_overlapping)
    trade_dates = exceedances["open_dates"]
    mod_asset_df = _build_asset_df(asset_df)

    trade_returns = _get_trade_returns(mod_asset_df, holding_period, trade_dates)
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
    trade_details = trade_details.to_dict("records")

    return {
        "summary_statistics": summary_statistics,
        "trade_details": trade_details,
    }
