"""Kalshi binary market orderbook.

YES + NO sides where YES_price + NO_price = 100 cents.
Handles both snapshot and delta updates from the WebSocket feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BookSide:
    """Single side of a Kalshi orderbook (yes or no)."""
    levels: dict[float, int] = field(default_factory=dict)

    def apply(self, price: float, delta: int) -> None:
        self.levels[price] = self.levels.get(price, 0) + delta
        if self.levels[price] <= 0:
            self.levels.pop(price, None)

    def set(self, price: float, size: int) -> None:
        """Absolute set (for snapshots)."""
        if size <= 0:
            self.levels.pop(price, None)
        else:
            self.levels[price] = size

    def best(self, reverse: bool = False) -> tuple[float, int] | None:
        if not self.levels:
            return None
        px = max(self.levels) if reverse else min(self.levels)
        return (px, self.levels[px])

    def clear(self) -> None:
        self.levels.clear()


@dataclass
class KalshiOrderBook:
    """Full orderbook for a single Kalshi market.

    Maintains YES and NO sides.  The implied YES ask is derived
    from the NO bid: yes_ask = 100 - no_best_bid.
    """
    yes: BookSide = field(default_factory=BookSide)
    no: BookSide = field(default_factory=BookSide)

    def apply_delta(self, side: str, price: float, delta: int) -> None:
        if side == "yes":
            self.yes.apply(price, delta)
        elif side == "no":
            self.no.apply(price, delta)

    def apply_snapshot(self, data: dict) -> None:
        """Apply a full snapshot from WebSocket or REST."""
        self.yes.clear()
        self.no.clear()
        for side_name, book_side in [("yes", self.yes), ("no", self.no)]:
            if side_name in data:
                for price, size in data[side_name]:
                    book_side.set(float(price), int(size))

    def top(self) -> tuple[
        tuple[float, int] | None,  # yes_bid (price, size)
        tuple[float, int] | None,  # yes_ask (price, size) — implied from NO
        float | None,              # mid
    ]:
        yes_bid = self.yes.best(reverse=True)

        # Implied yes ask from the NO side
        no_bid = self.no.best(reverse=True)
        yes_ask = (100.0 - no_bid[0], no_bid[1]) if no_bid else None

        mid = None
        if yes_bid and yes_ask:
            mid = (yes_bid[0] + yes_ask[0]) / 2.0

        return yes_bid, yes_ask, mid

    @property
    def yes_bid_price(self) -> float | None:
        b = self.yes.best(reverse=True)
        return b[0] if b else None

    @property
    def yes_ask_price(self) -> float | None:
        no_bid = self.no.best(reverse=True)
        return 100.0 - no_bid[0] if no_bid else None
