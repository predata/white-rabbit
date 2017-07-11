# pip-installed modules
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# project modules
from data.models import Signal
from prices.models import Asset

    
def get_signal_exceedance_dates(signal_id, alpha, rolling_window, holding_period, ignore_overlapping):
    signal = Signal.objects.get(pk=signal_id)
    signal_df = pd.DataFrame.from_records(signal.get_series().values("date", signal._value_accessor), index="date")
    signal_df.index = pd.to_datetime(signal_df.index)
    signal_df_rolling = signal_df.rolling(rolling_window)
    z_scores = (signal_df - signal_df_rolling.mean()) / signal_df_rolling.std()
    exceedances = z_scores.loc[z_scores[signal._value_accessor] > alpha]
    exceedance_dates = exceedances.index
    if ignore_overlapping:
        exceedance_dates = exceedance_dates[~(exceedance_dates.to_series().diff() < pd.Timedelta(days=holding_period))]
    trade_dates = exceedance_dates + BDay()
    close_dates = trade_dates + pd.DateOffset(days=holding_period) + (0 * BDay())

    return {
        "open_dates": trade_dates,
        "close_dates": close_dates
    }


def get_trade_returns(asset_id, holding_period, trade_dates):
    asset = Asset.objects.get(pk=asset_id)
    asset_df = pd.DataFrame.from_records(asset.prices.all().values("datetime", "price"), index="datetime",
                                         coerce_float=True).sort_index()
    cal_days_index = pd.date_range(asset_df.index.min(), asset_df.index.max())
    asset_df = asset_df.reindex(cal_days_index)
    asset_df = asset_df.fillna(method="bfill")
    asset_df.index = asset_df.index.tz_localize(None)
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


def get_trade_statistics(trade_returns_df, asset_df):
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
    for year in asset_df.index.year.unique():
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
