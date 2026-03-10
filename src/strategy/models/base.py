"""Base types for pluggable theoretical value models.

Every model receives a MarketState and returns P(spot finishes above open)
as a float in [0, 1].  Models are free to ignore fields they don't need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class MarketState:
    """Snapshot of all available market features at evaluation time.

    Core fields are always populated.  Optional fields may be None if
    the corresponding feature tracker is not running.
    """

    # -- Core (always available) --
    spot: float                          # current spot price
    spot_at_open: float                  # spot at 15-min window open
    vol_15m: float                       # realized vol scaled to 15 min
    time_remaining_sec: float            # seconds until settlement
    asset: str = ""                      # "BTC", "ETH", "SOL"

    # -- Kalshi book (available when Kalshi feed is running) --
    kalshi_bid: float | None = None      # yes_bid in cents
    kalshi_ask: float | None = None      # yes_ask in cents

    # -- Coinbase microstructure features --
    flow_imbalance: float | None = None  # (buy_vol - sell_vol) / total, rolling
    book_imbalance: float | None = None  # (bid_depth - ask_depth) / total, L2
    momentum_1m: float | None = None     # log return over last 60s

    @property
    def current_return(self) -> float:
        """Log return from window open to now."""
        if self.spot_at_open <= 0 or self.spot <= 0:
            return 0.0
        import math
        return math.log(self.spot / self.spot_at_open)

    @property
    def time_frac(self) -> float:
        """Time remaining as fraction of 15-min window."""
        return max(self.time_remaining_sec / 900.0, 0.0)

    @property
    def kalshi_mid(self) -> float | None:
        """Kalshi mid price in cents, or None."""
        if self.kalshi_bid is not None and self.kalshi_ask is not None:
            return (self.kalshi_bid + self.kalshi_ask) / 2
        return None

    @property
    def kalshi_spread(self) -> float | None:
        """Kalshi spread in cents, or None."""
        if self.kalshi_bid is not None and self.kalshi_ask is not None:
            return self.kalshi_ask - self.kalshi_bid
        return None


@runtime_checkable
class TheoModel(Protocol):
    """Interface that all fair-value models must implement."""

    name: str

    def fair_value(self, state: MarketState) -> float:
        """Return P(spot finishes above open) in [0, 1]."""
        ...
