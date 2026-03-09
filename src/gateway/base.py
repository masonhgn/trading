"""Abstract gateway interfaces.

Each exchange implements DataGateway (market data) and OrderGateway
(order entry) so that strategy and risk code stays exchange-agnostic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Callable


# -- Normalized market data events ------------------------------------------

@dataclass(slots=True)
class BookUpdate:
    """Normalized top-of-book update from any exchange."""
    exchange: str       # "coinbase" or "kalshi"
    symbol: str         # "BTC-USD" or "KXBTC15M-26MAR080345-45"
    bid: float          # best bid price
    ask: float          # best ask price
    bid_size: float     # size at best bid
    ask_size: float     # size at best ask
    mid: float          # (bid + ask) / 2
    ts: float           # epoch seconds


@dataclass(slots=True)
class TradeUpdate:
    """Normalized trade event from any exchange."""
    exchange: str
    symbol: str
    price: float
    size: float
    side: str           # "buy" or "sell"
    trade_id: str
    ts: float


@dataclass(slots=True)
class FillUpdate:
    """Normalized fill on one of our orders."""
    exchange: str
    order_id: str
    symbol: str
    side: str           # "buy" or "sell" / "yes" or "no"
    action: str         # "buy" or "sell"
    price: float        # fill price (cents for Kalshi, dollars for Coinbase)
    size: float         # filled quantity
    fee: float          # fee amount
    is_taker: bool
    ts: float


# -- Order request / response -----------------------------------------------

@dataclass(slots=True)
class OrderRequest:
    """Exchange-agnostic order request."""
    symbol: str
    side: str           # "buy" or "sell" for Coinbase; "yes" or "no" for Kalshi
    action: str         # "buy" or "sell" (Kalshi uses buy-yes / buy-no)
    size: int
    order_type: str     # "market" or "limit"
    price: float | None = None  # limit price (cents for Kalshi)
    client_order_id: str = ""


@dataclass(slots=True)
class OrderResponse:
    """Result of an order submission."""
    success: bool
    order_id: str = ""
    error: str = ""


@dataclass(slots=True)
class PositionInfo:
    """Current position on a single instrument."""
    symbol: str
    position: int           # net contracts (positive = long)
    realized_pnl: float     # in cents
    fees_paid: float        # in cents


@dataclass(slots=True)
class OrderUpdate:
    """Real-time order status update from exchange WebSocket."""
    exchange: str
    order_id: str
    symbol: str
    side: str               # "yes" or "no"
    price: float            # yes_price in cents
    initial_count: int
    remaining_count: int
    fill_count: int
    status: str             # "resting", "canceled", "executed"
    ts: float


@dataclass(slots=True)
class PositionUpdate:
    """Real-time position update from exchange WebSocket."""
    exchange: str
    symbol: str
    position: int
    realized_pnl: float     # in cents
    fees_paid: float        # in cents
    ts: float


# -- Abstract gateways -------------------------------------------------------

class DataGateway(ABC):
    """Market data feed for a single exchange.

    Implementations connect via WebSocket and/or REST, normalize
    messages, and yield them through async iterators.
    """

    @property
    @abstractmethod
    def exchange_name(self) -> str: ...

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection(s) to the exchange."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect."""

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for the given symbols."""

    @abstractmethod
    def on_book_update(self, callback: Callable[[BookUpdate], None]) -> None:
        """Register a callback for book updates."""

    @abstractmethod
    def on_trade(self, callback: Callable[[TradeUpdate], None]) -> None:
        """Register a callback for trade events."""

    @abstractmethod
    async def run(self) -> None:
        """Run the data feed loop (blocking)."""


class OrderGateway(ABC):
    """Order entry gateway for a single exchange.

    Wraps exchange-specific REST/WS order APIs behind a uniform
    interface.  All blocking calls are run in an executor.
    """

    @property
    @abstractmethod
    def exchange_name(self) -> str: ...

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection / auth."""

    @abstractmethod
    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        """Submit an order. Returns immediately with order ID or error."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order. Returns True on success."""

    @abstractmethod
    async def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all orders (optionally for a symbol). Returns count cancelled."""

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[PositionInfo]:
        """Query current positions."""

    @abstractmethod
    async def get_balance(self) -> float:
        """Get available balance (cents for Kalshi, USD for Coinbase)."""

    def on_fill(self, callback: Callable[[FillUpdate], None]) -> None:
        """Register a callback for fill events (optional, WS-based)."""
        pass
