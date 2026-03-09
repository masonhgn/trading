"""Shared test fixtures for the trading system."""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gateway.base import BookUpdate, FillUpdate


def make_fill(
    *,
    exchange: str = "kalshi",
    order_id: str = "order-001",
    symbol: str = "TEST-SYMBOL",
    side: str = "yes",
    action: str = "buy",
    price: float = 50.0,
    size: float = 1.0,
    fee: float = 0.0,
    is_taker: bool = False,
    ts: float | None = None,
) -> FillUpdate:
    """Factory for FillUpdate with sensible defaults."""
    return FillUpdate(
        exchange=exchange,
        order_id=order_id,
        symbol=symbol,
        side=side,
        action=action,
        price=price,
        size=size,
        fee=fee,
        is_taker=is_taker,
        ts=ts if ts is not None else time.time(),
    )


def make_book_update(
    *,
    exchange: str = "kalshi",
    symbol: str = "TEST-SYMBOL",
    bid: float = 49.0,
    ask: float = 51.0,
    bid_size: float = 10.0,
    ask_size: float = 10.0,
    mid: float | None = None,
    ts: float | None = None,
) -> BookUpdate:
    """Factory for BookUpdate with sensible defaults."""
    return BookUpdate(
        exchange=exchange,
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        mid=mid if mid is not None else (bid + ask) / 2,
        ts=ts if ts is not None else time.time(),
    )


@pytest.fixture
def fill_factory():
    """Fixture exposing make_fill for tests that prefer fixture injection."""
    return make_fill


@pytest.fixture
def book_update_factory():
    """Fixture exposing make_book_update for tests that prefer fixture injection."""
    return make_book_update
