"""Coinbase L2 order book manager.

Maintains a local copy of the order book from WebSocket updates
and produces snapshots on demand.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class L2OrderBook:
    """Single-symbol Level 2 order book.

    Thread-safe: Coinbase WS callbacks run in a background thread,
    while reads happen on the asyncio event loop thread.
    """

    symbol: str = ""
    _bids: dict[float, float] = field(default_factory=lambda: defaultdict(float))
    _asks: dict[float, float] = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def apply_update(self, side: str, price: str | float, size: str | float) -> None:
        p = float(price)
        s = float(size)
        with self._lock:
            book = self._bids if side == "bid" else self._asks
            if s <= 0:
                book.pop(p, None)
            else:
                book[p] = s

    def best_bid(self) -> tuple[float, float] | None:
        with self._lock:
            if not self._bids:
                return None
            p = max(self._bids)
            return (p, self._bids[p])

    def best_ask(self) -> tuple[float, float] | None:
        with self._lock:
            if not self._asks:
                return None
            p = min(self._asks)
            return (p, self._asks[p])

    def mid(self) -> float | None:
        with self._lock:
            if not self._bids or not self._asks:
                return None
            return (max(self._bids) + min(self._asks)) / 2

    def top_levels(self, depth: int = 20) -> tuple[list, list]:
        with self._lock:
            bids = sorted(self._bids.items(), reverse=True)[:depth]
            asks = sorted(self._asks.items())[:depth]
        return bids, asks


class OrderBookManager:
    """Manages L2 books for multiple symbols."""

    def __init__(self) -> None:
        self._books: dict[str, L2OrderBook] = {}

    def get_or_create(self, symbol: str) -> L2OrderBook:
        if symbol not in self._books:
            self._books[symbol] = L2OrderBook(symbol=symbol)
        return self._books[symbol]

    def get(self, symbol: str) -> L2OrderBook | None:
        return self._books.get(symbol)
