"""Fair value model for Kalshi 15-min crypto prediction contracts.

Computes theoretical probability that spot will finish higher than its
opening price at the start of the 15-minute window, using a simple
normal CDF model calibrated with recent realized volatility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import norm


# Kalshi ticker times are in US Eastern (UTC-4 EDT / UTC-5 EST).
# Our data is from March 8, 2026 (still EST before spring forward on Mar 9),
# but empirically the tickers use UTC-4 offset based on observation timestamps.
EDT_OFFSET = timedelta(hours=-4)


@dataclass(slots=True)
class ContractInfo:
    ticker: str
    asset: str           # "BTC", "ETH", "SOL"
    cb_symbol: str       # "BTC-USD", "ETH-USD", "SOL-USD"
    series: str          # "KXBTC15M"
    window_start: float  # epoch seconds UTC
    window_end: float    # epoch seconds UTC


def parse_contract_ticker(ticker: str) -> ContractInfo | None:
    """Parse a Kalshi 15-min ticker into its components.

    Format: KXBTC15M-26MAR080345-45
             series   date+time  min
    """
    m = re.match(r"(KX(\w+)15M)-(\d{2})([A-Z]{3})(\d{2})(\d{4})-(\d{2})", ticker)
    if not m:
        return None

    series = m.group(1)
    asset = m.group(2)
    year_short = int(m.group(3))  # "26" -> 2026
    month_str = m.group(4)        # "MAR"
    day = int(m.group(5))         # "08" -> 8
    time_str = m.group(6)         # "0345"
    # minute suffix m.group(7) is redundant

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = month_map.get(month_str)
    if not month:
        return None

    year = 2000 + year_short
    hour = int(time_str[:2])
    minute = int(time_str[2:])

    # Window end in local time (EDT), convert to UTC
    local_end = datetime(year, month, day, hour, minute, tzinfo=timezone(EDT_OFFSET))
    window_end_utc = local_end.astimezone(timezone.utc)
    window_start_utc = window_end_utc - timedelta(minutes=15)

    cb_symbol_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}

    return ContractInfo(
        ticker=ticker,
        asset=asset,
        cb_symbol=cb_symbol_map.get(asset, f"{asset}-USD"),
        series=series,
        window_start=window_start_utc.timestamp(),
        window_end=window_end_utc.timestamp(),
    )


def compute_fair_value(
    spot_now: float,
    spot_at_open: float,
    vol_15m: float,
    time_remaining_sec: float,
) -> float:
    """Compute fair probability that spot finishes above spot_at_open.

    Args:
        spot_now: Current spot price.
        spot_at_open: Spot price at the start of the 15-min window.
        vol_15m: Standard deviation of 15-minute log returns.
        time_remaining_sec: Seconds until contract settlement.

    Returns:
        Probability in [0, 1] (multiply by 100 for cents).
    """
    if time_remaining_sec <= 0:
        return 1.0 if spot_now > spot_at_open else 0.0

    if vol_15m <= 0 or spot_at_open <= 0:
        return 0.5

    # Current log return from open
    current_return = np.log(spot_now / spot_at_open)

    # Time fraction remaining (of the full 15 min window)
    T = time_remaining_sec / (15 * 60)
    T = max(T, 1e-6)  # avoid division by zero

    # Under GBM, the remaining return is N(0, vol^2 * T)
    # P(S_end > S_open) = P(current_return + remaining_return > 0)
    # = P(remaining_return > -current_return)
    # = Phi(current_return / (vol * sqrt(T)))
    z = current_return / (vol_15m * np.sqrt(T))

    return float(norm.cdf(z))


class VolEstimator:
    """Rolling realized volatility estimator for 15-minute returns."""

    # Per-asset vol floors based on typical annualized vol scaled to 15-min.
    # Prevents the model from being overconfident during quiet periods.
    # BTC ~60% ann → 0.0032, ETH ~80% → 0.0043, SOL ~100% → 0.0053
    VOL_FLOOR = {
        "BTC": 0.0030,
        "ETH": 0.0040,
        "SOL": 0.0050,
    }
    DEFAULT_VOL_FLOOR = 0.0035

    def __init__(self, lookback_sec: float = 600.0, asset: str = "") -> None:
        from collections import deque
        self._lookback = lookback_sec
        self._vol_floor = self.VOL_FLOOR.get(asset, self.DEFAULT_VOL_FLOOR)
        self._prices: deque[tuple[float, float]] = deque()  # (ts, price)

    def update(self, ts: float, price: float) -> None:
        self._prices.append((ts, price))
        # Evict old prices — O(1) popleft with deque
        cutoff = ts - self._lookback
        while self._prices and self._prices[0][0] < cutoff:
            self._prices.popleft()

    def vol_15m(self) -> float:
        """Estimate 15-minute realized vol from recent price history."""
        if len(self._prices) < 20:
            return 0.001  # default low vol

        prices = np.array([p[1] for p in self._prices])
        timestamps = np.array([p[0] for p in self._prices])

        # Log returns
        log_rets = np.diff(np.log(prices))
        if len(log_rets) == 0:
            return 0.001

        # Average time step
        dt_avg = np.mean(np.diff(timestamps))
        if dt_avg <= 0:
            return 0.001

        # Variance per second, then scale to 15 minutes
        var_per_sec = np.var(log_rets) / dt_avg
        vol_15m = np.sqrt(var_per_sec * 15 * 60)

        return max(vol_15m, self._vol_floor)

    def reset(self) -> None:
        self._prices.clear()
