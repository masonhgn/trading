"""Tests for OrderManager — order placement, cancellation, quoting, and fills."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import AsyncMock, PropertyMock

import pytest

from conftest import make_fill
from gateway.base import FillUpdate, OrderGateway, OrderRequest, OrderResponse
from risk.order_manager import ActiveOrder, OrderManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gateway() -> AsyncMock:
    """AsyncMock gateway implementing the OrderGateway interface."""
    gw = AsyncMock(spec=OrderGateway)
    type(gw).exchange_name = PropertyMock(return_value="test")
    gw.submit_order.return_value = OrderResponse(success=True, order_id="test-123")
    gw.cancel_order.return_value = True
    return gw


@pytest.fixture
def om(mock_gateway: AsyncMock) -> OrderManager:
    """OrderManager wired to the mock gateway."""
    return OrderManager(gateway=mock_gateway, symbol="TEST-SYM", default_size=1)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

async def test_place_bid_order(om: OrderManager, mock_gateway: AsyncMock):
    result = await om.place_order("bid", price=55)

    assert result is not None
    assert isinstance(result, ActiveOrder)
    assert result.order_id == "test-123"
    assert result.side == "bid"
    assert result.price == 55
    assert om.active_bid is result

    # Gateway should have received a buy-YES request at 55c
    req: OrderRequest = mock_gateway.submit_order.call_args[0][0]
    assert req.side == "yes"
    assert req.action == "buy"
    assert req.price == 55


async def test_place_ask_order(om: OrderManager, mock_gateway: AsyncMock):
    result = await om.place_order("ask", price=60)

    assert result is not None
    assert result.side == "ask"
    assert om.active_ask is result

    # Ask => buy-NO with price = round(100 - 60) = 40
    req: OrderRequest = mock_gateway.submit_order.call_args[0][0]
    assert req.side == "no"
    assert req.action == "buy"
    assert req.price == 40


# ---------------------------------------------------------------------------
# Duplicate-order guard
# ---------------------------------------------------------------------------

async def test_duplicate_order_guard_bid(om: OrderManager, mock_gateway: AsyncMock):
    await om.place_order("bid", price=50)
    mock_gateway.submit_order.reset_mock()

    result = await om.place_order("bid", price=50)

    assert result is None
    mock_gateway.submit_order.assert_not_called()


async def test_duplicate_order_guard_ask(om: OrderManager, mock_gateway: AsyncMock):
    await om.place_order("ask", price=50)
    mock_gateway.submit_order.reset_mock()

    result = await om.place_order("ask", price=50)

    assert result is None
    mock_gateway.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# Price validation
# ---------------------------------------------------------------------------

async def test_invalid_price_rejected_low(om: OrderManager, mock_gateway: AsyncMock):
    # bid price=0 => req.price = round(0) = 0 < 1 => rejected
    result = await om.place_order("bid", price=0)

    assert result is None
    mock_gateway.submit_order.assert_not_called()


async def test_invalid_price_rejected_high(om: OrderManager, mock_gateway: AsyncMock):
    # ask price=0 => req.price = round(100 - 0) = 100 > 99 => rejected
    result = await om.place_order("ask", price=0)

    assert result is None
    mock_gateway.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# Gateway failure
# ---------------------------------------------------------------------------

async def test_gateway_failure_returns_none(om: OrderManager, mock_gateway: AsyncMock):
    mock_gateway.submit_order.return_value = OrderResponse(
        success=False, error="exchange down"
    )

    result = await om.place_order("bid", price=50)

    assert result is None
    assert om.active_bid is None


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

async def test_cancel_order_success(om: OrderManager, mock_gateway: AsyncMock):
    await om.place_order("bid", price=50)

    cancelled = await om.cancel_order("bid")

    assert cancelled is True
    assert om.active_bid is None
    mock_gateway.cancel_order.assert_awaited_once_with("test-123")


async def test_cancel_nonexistent_returns_true(om: OrderManager, mock_gateway: AsyncMock):
    result = await om.cancel_order("bid")

    assert result is True
    mock_gateway.cancel_order.assert_not_called()


async def test_cancel_all(om: OrderManager, mock_gateway: AsyncMock):
    mock_gateway.submit_order.side_effect = [
        OrderResponse(success=True, order_id="bid-1"),
        OrderResponse(success=True, order_id="ask-1"),
    ]
    await om.place_order("bid", price=45)
    await om.place_order("ask", price=55)

    await om.cancel_all()

    assert om.active_bid is None
    assert om.active_ask is None
    assert mock_gateway.cancel_order.await_count == 2


# ---------------------------------------------------------------------------
# Quote updates (cancel-replace)
# ---------------------------------------------------------------------------

async def test_update_quote_same_price_skips(om: OrderManager, mock_gateway: AsyncMock):
    original = await om.place_order("bid", price=50)
    mock_gateway.submit_order.reset_mock()
    mock_gateway.cancel_order.reset_mock()

    result = await om.update_quote("bid", price=50)

    assert result is original
    mock_gateway.cancel_order.assert_not_called()
    mock_gateway.submit_order.assert_not_called()


async def test_update_quote_different_price(om: OrderManager, mock_gateway: AsyncMock):
    await om.place_order("bid", price=50)

    mock_gateway.submit_order.return_value = OrderResponse(
        success=True, order_id="test-456"
    )

    result = await om.update_quote("bid", price=55)

    assert result is not None
    assert result.price == 55
    assert result.order_id == "test-456"
    mock_gateway.cancel_order.assert_awaited_once()


async def test_update_quote_cancel_fails(om: OrderManager, mock_gateway: AsyncMock):
    await om.place_order("bid", price=50)
    mock_gateway.cancel_order.return_value = False

    result = await om.update_quote("bid", price=55)

    assert result is None
    # Original order should still be tracked (cancel failed)
    assert om.active_bid is not None
    assert om.active_bid.price == 50


# ---------------------------------------------------------------------------
# Fill processing
# ---------------------------------------------------------------------------

async def test_process_fill_matches_bid(om: OrderManager):
    await om.place_order("bid", price=50)

    fill = make_fill(order_id="test-123", side="yes", price=50)
    side = om.process_fill(fill)

    assert side == "bid"
    assert om.active_bid is None


async def test_process_fill_matches_ask(om: OrderManager):
    await om.place_order("ask", price=60)

    fill = make_fill(order_id="test-123", side="no", price=40)
    side = om.process_fill(fill)

    assert side == "ask"
    assert om.active_ask is None


async def test_process_fill_unmatched(om: OrderManager):
    fill = make_fill(order_id="unknown-999", side="yes", price=50)
    side = om.process_fill(fill)

    assert side is None


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

async def test_order_count_increments(om: OrderManager, mock_gateway: AsyncMock):
    assert om.order_count == 0

    await om.place_order("bid", price=50)
    assert om.order_count == 1

    mock_gateway.submit_order.return_value = OrderResponse(
        success=True, order_id="test-456"
    )
    await om.place_order("ask", price=60)
    assert om.order_count == 2
