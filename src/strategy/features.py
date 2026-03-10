"""Streaming feature trackers for microstructure signals.

Each tracker maintains rolling state and exposes a current value.
They are designed to be updated on every tick/trade and queried
when building a MarketState for model evaluation.
"""

from __future__ import annotations

from collections import deque


class FlowTracker:
    """Rolling order flow imbalance from Coinbase trades.

    Tracks (buy_volume - sell_volume) / total_volume over a
    configurable lookback window.
    """

    def __init__(self, lookback_sec: float = 120.0) -> None:
        self._lookback = lookback_sec
        # (ts, signed_volume): positive = buy, negative = sell
        self._trades: deque[tuple[float, float]] = deque()
        self._buy_vol = 0.0
        self._sell_vol = 0.0

    def update(self, ts: float, size: float, side: str) -> None:
        """Record a trade. side is "buy" or "sell"."""
        if side == "buy":
            self._trades.append((ts, size))
            self._buy_vol += size
        else:
            self._trades.append((ts, -size))
            self._sell_vol += size
        self._evict(ts)

    def _evict(self, now: float) -> None:
        cutoff = now - self._lookback
        while self._trades and self._trades[0][0] < cutoff:
            _, vol = self._trades.popleft()
            if vol > 0:
                self._buy_vol -= vol
            else:
                self._sell_vol -= (-vol)

    @property
    def imbalance(self) -> float | None:
        """Current flow imbalance in [-1, 1], or None if no data."""
        total = self._buy_vol + self._sell_vol
        if total <= 0:
            return None
        return (self._buy_vol - self._sell_vol) / total

    def reset(self) -> None:
        self._trades.clear()
        self._buy_vol = 0.0
        self._sell_vol = 0.0


class BookImbalanceTracker:
    """L2 order book imbalance from Coinbase.

    Computes (bid_depth - ask_depth) / (bid_depth + ask_depth)
    from the most recent book snapshot.
    """

    def __init__(self) -> None:
        self._bid_depth = 0.0
        self._ask_depth = 0.0

    def update(self, bid_depth: float, ask_depth: float) -> None:
        """Update with total depth on each side (in base currency)."""
        self._bid_depth = bid_depth
        self._ask_depth = ask_depth

    @property
    def imbalance(self) -> float | None:
        """Current book imbalance in [-1, 1], or None if no data."""
        total = self._bid_depth + self._ask_depth
        if total <= 0:
            return None
        return (self._bid_depth - self._ask_depth) / total

    def reset(self) -> None:
        self._bid_depth = 0.0
        self._ask_depth = 0.0


class MomentumTracker:
    """Short-term price momentum (log return over a lookback window)."""

    def __init__(self, lookback_sec: float = 60.0) -> None:
        self._lookback = lookback_sec
        self._prices: deque[tuple[float, float]] = deque()

    def update(self, ts: float, price: float) -> None:
        self._prices.append((ts, price))
        cutoff = ts - self._lookback * 2  # keep 2x buffer for interpolation
        while self._prices and self._prices[0][0] < cutoff:
            self._prices.popleft()

    @property
    def momentum(self) -> float | None:
        """Log return over the lookback window, or None if insufficient data."""
        if len(self._prices) < 2:
            return None

        now_ts = self._prices[-1][0]
        now_price = self._prices[-1][1]
        target_ts = now_ts - self._lookback

        # Find the price closest to target_ts
        best_price = None
        best_dist = float("inf")
        for ts, price in self._prices:
            dist = abs(ts - target_ts)
            if dist < best_dist:
                best_dist = dist
                best_price = price

        if best_price is None or best_price <= 0 or now_price <= 0:
            return None

        import math
        return math.log(now_price / best_price)

    def reset(self) -> None:
        self._prices.clear()
