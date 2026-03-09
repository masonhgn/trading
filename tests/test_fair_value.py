"""Tests for fair value model, parse_contract_ticker, and VolEstimator."""

import time

import numpy as np
import pytest

from strategy.fair_value import (
    ContractInfo,
    VolEstimator,
    compute_fair_value,
    parse_contract_ticker,
)


# ── parse_contract_ticker ────────────────────────────────────────────


class TestParseContractTicker:
    def test_parse_btc_ticker(self):
        info = parse_contract_ticker("KXBTC15M-26MAR080345-45")
        assert info is not None
        assert info.asset == "BTC"
        assert info.cb_symbol == "BTC-USD"
        assert info.series == "KXBTC15M"
        # Window end is 03:45 EDT => 07:45 UTC on 2026-03-08
        # Window start is 15 min earlier => 07:30 UTC
        assert info.window_end - info.window_start == 900

    def test_parse_eth_ticker(self):
        info = parse_contract_ticker("KXETH15M-26MAR080345-45")
        assert info is not None
        assert info.asset == "ETH"
        assert info.cb_symbol == "ETH-USD"
        assert info.series == "KXETH15M"

    def test_parse_sol_ticker(self):
        info = parse_contract_ticker("KXSOL15M-26MAR080345-45")
        assert info is not None
        assert info.asset == "SOL"
        assert info.cb_symbol == "SOL-USD"
        assert info.series == "KXSOL15M"

    def test_parse_invalid_ticker(self):
        assert parse_contract_ticker("INVALID-TICKER") is None
        assert parse_contract_ticker("") is None
        assert parse_contract_ticker("KXBTC15M-bad") is None

    def test_window_times_15min(self):
        info = parse_contract_ticker("KXBTC15M-26MAR081200-00")
        assert info is not None
        assert info.window_end - info.window_start == 900


# ── compute_fair_value ───────────────────────────────────────────────


class TestComputeFairValue:
    """Tests for compute_fair_value probability model."""

    VOL = 0.005  # typical 15-min vol
    WINDOW = 15 * 60  # full 15-min window in seconds

    def test_fv_spot_above_open(self):
        fv = compute_fair_value(
            spot_now=100.5, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=self.WINDOW / 2,
        )
        assert fv > 0.5

    def test_fv_spot_below_open(self):
        fv = compute_fair_value(
            spot_now=99.5, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=self.WINDOW / 2,
        )
        assert fv < 0.5

    def test_fv_spot_equals_open(self):
        fv = compute_fair_value(
            spot_now=100.0, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=self.WINDOW / 2,
        )
        assert fv == pytest.approx(0.5)

    def test_fv_expired_above(self):
        fv = compute_fair_value(
            spot_now=101.0, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=0,
        )
        assert fv == 1.0

    def test_fv_expired_below(self):
        fv = compute_fair_value(
            spot_now=99.0, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=0,
        )
        assert fv == 0.0

    def test_fv_zero_vol(self):
        fv = compute_fair_value(
            spot_now=101.0, spot_at_open=100.0,
            vol_15m=0.0, time_remaining_sec=self.WINDOW / 2,
        )
        assert fv == 0.5

    def test_fv_zero_open(self):
        fv = compute_fair_value(
            spot_now=101.0, spot_at_open=0.0,
            vol_15m=self.VOL, time_remaining_sec=self.WINDOW / 2,
        )
        assert fv == 0.5

    def test_fv_approaches_1_far_above(self):
        fv = compute_fair_value(
            spot_now=110.0, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=60,
        )
        assert fv > 0.99

    def test_fv_approaches_0_far_below(self):
        fv = compute_fair_value(
            spot_now=90.0, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=60,
        )
        assert fv < 0.01

    def test_fv_high_vol_moderate_return(self):
        """High vol with moderate return should stay closer to 0.5."""
        fv_high_vol = compute_fair_value(
            spot_now=100.2, spot_at_open=100.0,
            vol_15m=0.02, time_remaining_sec=self.WINDOW / 2,
        )
        fv_low_vol = compute_fair_value(
            spot_now=100.2, spot_at_open=100.0,
            vol_15m=0.002, time_remaining_sec=self.WINDOW / 2,
        )
        # High vol => closer to 0.5 than low vol
        assert abs(fv_high_vol - 0.5) < abs(fv_low_vol - 0.5)

    def test_fv_low_vol_moderate_return(self):
        """Low vol with moderate return should push further from 0.5."""
        fv = compute_fair_value(
            spot_now=100.3, spot_at_open=100.0,
            vol_15m=0.001, time_remaining_sec=self.WINDOW / 2,
        )
        assert fv > 0.9

    def test_fv_time_decay(self):
        """More time remaining => fair value stays closer to 0.5."""
        fv_more_time = compute_fair_value(
            spot_now=100.2, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=self.WINDOW,
        )
        fv_less_time = compute_fair_value(
            spot_now=100.2, spot_at_open=100.0,
            vol_15m=self.VOL, time_remaining_sec=60,
        )
        assert abs(fv_more_time - 0.5) < abs(fv_less_time - 0.5)


# ── VolEstimator ─────────────────────────────────────────────────────


class TestVolEstimator:
    def test_vol_estimator_default(self):
        """Fewer than 20 prices returns the default low vol."""
        ve = VolEstimator()
        # Add only 10 prices
        for i in range(10):
            ve.update(1000.0 + i, 100.0 + i * 0.01)
        assert ve.vol_15m() == 0.001

    def test_vol_estimator_with_data(self):
        """With enough data the estimator returns a positive vol > default."""
        ve = VolEstimator(lookback_sec=600.0)
        base = 50_000.0  # BTC-ish price
        np.random.seed(42)
        for i in range(100):
            ts = 1000.0 + i * 5  # 5-second ticks
            price = base * np.exp(np.random.normal(0, 0.0002))
            ve.update(ts, price)
        vol = ve.vol_15m()
        assert vol > 0.001  # should exceed default
        assert vol < 1.0    # sanity upper bound

    def test_vol_estimator_eviction(self):
        """Old prices outside the lookback window should be evicted."""
        ve = VolEstimator(lookback_sec=100.0)
        # Insert 30 prices spanning 0..145 sec (> lookback)
        for i in range(30):
            ve.update(1000.0 + i * 5, 100.0)
        # The oldest entries should have been evicted
        assert len(ve._prices) < 30
        # Only prices with ts >= (last_ts - lookback) should remain
        last_ts = ve._prices[-1][0]
        for ts, _ in ve._prices:
            assert ts >= last_ts - 100.0

    def test_vol_estimator_reset(self):
        """reset() should clear all stored data."""
        ve = VolEstimator()
        for i in range(25):
            ve.update(1000.0 + i, 100.0 + i * 0.01)
        assert len(ve._prices) > 0
        ve.reset()
        assert len(ve._prices) == 0
        assert ve.vol_15m() == 0.001
