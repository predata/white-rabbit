import unittest

import numpy as np
import pandas as pd

import constants
import white_rabbit.vectorized_backtest as vectorized_backtest


class VectorizedBacktestTestCase(unittest.TestCase):
    def setUp(self):
        self.signal_df = constants.SIGNAL_DF
        self.value_accessor = "level"
        self.asset_df = constants.ASSET_DF
        self.alpha = 1.5
        self.rolling_window = 100
        self.holding_period = 10
        self.holding_periods = range(1, 11)
        self.ignore_overlapping = True
        self.is_buy = False
        self.percentiles = [25, 50, 75]

    def tearDown(self):
        pass

    def test_get_trade_statistics(self):
        trade_statistics = vectorized_backtest.get_trade_statistics(
            self.signal_df,
            self.value_accessor,
            self.asset_df,
            self.alpha,
            self.rolling_window,
            self.holding_period,
            self.ignore_overlapping,
            self.is_buy
        )
        np.testing.assert_equal(trade_statistics, constants.TRADE_STATISTICS)

    def test_get_trade_percentiles(self):
        trade_percentiles = vectorized_backtest.get_trade_percentiles(
            self.signal_df,
            self.value_accessor,
            self.asset_df,
            self.alpha,
            self.rolling_window,
            self.holding_periods,
            self.ignore_overlapping,
            self.is_buy,
            self.percentiles
        )
        np.testing.assert_equal(trade_percentiles, constants.TRADE_PERCENTILES)

    def test_get_trade_returns(self):
        trade_returns = vectorized_backtest.get_trade_returns(
            self.signal_df,
            self.value_accessor,
            self.asset_df,
            self.alpha,
            self.rolling_window,
            self.holding_period,
            self.ignore_overlapping,
            self.is_buy
        )
        trade_returns_keys = sorted(trade_returns.keys())
        desired_trade_returns_keys = sorted(constants.TRADE_RETURNS.keys())
        self.assertEqual(trade_returns_keys, desired_trade_returns_keys)

        for key in desired_trade_returns_keys:
            pd.testing.assert_series_equal(trade_returns[key], constants.TRADE_RETURNS[key], obj=key)
