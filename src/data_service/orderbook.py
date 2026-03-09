"""In-memory L2 order book maintained from WebSocket incremental updates.

Maintains a sorted book per symbol. Designed for speed:
- Uses sorted dict (SortedDict from sortedcontainers would be ideal,
  but we use a plain dict + sorted keys to avoid the dependency).
- Publishes snapshots after each update batch.
"""

from __future__ import annotations

import threading
import time
from common.models import OrderBookLevel, OrderBookSnapshot


class L2OrderBook:
    """Single-symbol Level 2 order book. Thread-safe."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._bids: dict[float, float] = {}  # price -> size
        self._asks: dict[float, float] = {}  # price -> size
        self._lock = threading.Lock()

    def apply_snapshot(self, bids: list[list], asks: list[list]) -> None:
        """Replace the entire book with a snapshot."""
        with self._lock:
            self._bids.clear()
            self._asks.clear()
            for price, size in bids:
                p, s = float(price), float(size)
                if s > 0:
                    self._bids[p] = s
            for price, size in asks:
                p, s = float(price), float(size)
                if s > 0:
                    self._asks[p] = s

    def apply_update(self, side: str, price: str, size: str) -> None:
        """Apply a single L2 update. size=0 means remove the level."""
        p, s = float(price), float(size)
        with self._lock:
            book = self._bids if side == "bid" else self._asks
            if s == 0:
                book.pop(p, None)
            else:
                book[p] = s

    def snapshot(self, depth: int = 20, exchange_ts: float = 0.0) -> OrderBookSnapshot:
        """Return a snapshot of the top N levels."""
        with self._lock:
            sorted_bids = sorted(self._bids.items(), key=lambda x: -x[0])[:depth]
            sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])[:depth]
        return OrderBookSnapshot(
            symbol=self.symbol,
            bids=[OrderBookLevel(p, s) for p, s in sorted_bids],
            asks=[OrderBookLevel(p, s) for p, s in sorted_asks],
            exchange_ts=exchange_ts,
            local_ts=time.monotonic(),
        )

    @property
    def best_bid(self) -> float | None:
        with self._lock:
            return max(self._bids.keys()) if self._bids else None

    @property
    def best_ask(self) -> float | None:
        with self._lock:
            return min(self._asks.keys()) if self._asks else None


class OrderBookManager:
    """Manages L2 books for multiple symbols."""

    def __init__(self) -> None:
        self._books: dict[str, L2OrderBook] = {}

    def get_or_create(self, symbol: str) -> L2OrderBook:
        if symbol not in self._books:
            self._books[symbol] = L2OrderBook(symbol)
        return self._books[symbol]

    def get(self, symbol: str) -> L2OrderBook | None:
        return self._books.get(symbol)

    @property
    def symbols(self) -> list[str]:
        return list(self._books.keys())
