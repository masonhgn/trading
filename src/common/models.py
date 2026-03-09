"""Shared data models for inter-service messages.

All models serialize to plain dicts for msgpack transport.
Using dataclasses for zero-overhead field access on the hot path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass(slots=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(slots=True)
class OrderBookSnapshot:
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    exchange_ts: float  # exchange timestamp (seconds)
    local_ts: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bids": [[l.price, l.size] for l in self.bids],
            "asks": [[l.price, l.size] for l in self.asks],
            "exchange_ts": self.exchange_ts,
            "local_ts": self.local_ts,
        }

    @staticmethod
    def from_dict(d: dict) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            symbol=d["symbol"],
            bids=[OrderBookLevel(p, s) for p, s in d["bids"]],
            asks=[OrderBookLevel(p, s) for p, s in d["asks"]],
            exchange_ts=d["exchange_ts"],
            local_ts=d.get("local_ts", 0.0),
        )

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass(slots=True)
class Trade:
    symbol: str
    price: float
    size: float
    side: Literal["buy", "sell"]
    trade_id: str
    exchange_ts: float
    local_ts: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> Trade:
        return Trade(**d)


@dataclass(slots=True)
class Ticker:
    symbol: str
    price: float
    volume_24h: float
    low_24h: float
    high_24h: float
    best_bid: float
    best_ask: float
    exchange_ts: float
    local_ts: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> Ticker:
        return Ticker(**d)


# ── Execution models ────────────────────────────────────────────────


@dataclass(slots=True)
class OrderRequest:
    """A signal from strategy requesting an order placement."""
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    order_type: Literal["market", "limit"] = "limit"
    limit_price: float | None = None
    client_order_id: str = ""
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> OrderRequest:
        return OrderRequest(**d)


@dataclass(slots=True)
class OrderStatus:
    """Current state of an order tracked by the execution service."""
    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    limit_price: float | None
    size: float
    filled_size: float
    avg_fill_price: float
    status: Literal["pending", "open", "partially_filled", "filled", "cancelled", "rejected"]
    reject_reason: str = ""
    local_ts: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> OrderStatus:
        return OrderStatus(**d)


@dataclass(slots=True)
class Fill:
    """A single fill (execution) on an order."""
    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float
    fee: float
    trade_id: str
    local_ts: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> Fill:
        return Fill(**d)


# ── Kalshi models ──────────────────────────────────────────────────


@dataclass(slots=True)
class KalshiMarket:
    """A Kalshi prediction market contract."""
    ticker: str           # e.g. "KXBTC15M-26MAR08T1200"
    event_ticker: str     # e.g. "KXBTC15M-26MAR08T1200"
    series_ticker: str    # e.g. "KXBTC15M"
    title: str
    subtitle: str
    status: str           # "open", "closed", "settled"
    yes_bid: float        # best yes bid in cents (0-100)
    yes_ask: float        # best yes ask in cents (0-100)
    last_price: float     # last trade price in cents
    volume: int           # number of contracts traded
    open_interest: int
    expiration_ts: float  # epoch seconds
    local_ts: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> KalshiMarket:
        return KalshiMarket(**d)

    @property
    def mid(self) -> float | None:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_bid + self.yes_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return self.yes_ask - self.yes_bid
        return None
