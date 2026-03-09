"""Tests for the execution service (order manager and models)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.models import OrderRequest, OrderStatus, Fill
from execution_service.order_manager import OrderManager, generate_client_order_id


def test_generate_client_order_id():
    id1 = generate_client_order_id()
    id2 = generate_client_order_id()
    assert len(id1) == 16
    assert id1 != id2


def test_register_order():
    om = OrderManager()
    req = OrderRequest(
        symbol="BTC-USD",
        side="buy",
        size=0.01,
        order_type="limit",
        limit_price=50000.0,
        client_order_id="test-001",
    )
    status = om.register(req)
    assert status.client_order_id == "test-001"
    assert status.symbol == "BTC-USD"
    assert status.status == "pending"
    assert status.filled_size == 0.0
    assert status.size == 0.01


def test_update_exchange_id():
    om = OrderManager()
    req = OrderRequest(symbol="ETH-USD", side="sell", size=1.0, client_order_id="test-002")
    om.register(req)

    result = om.update_exchange_id("test-002", "exchange-abc")
    assert result.exchange_order_id == "exchange-abc"
    assert result.status == "open"


def test_apply_fill_partial():
    om = OrderManager()
    req = OrderRequest(symbol="BTC-USD", side="buy", size=1.0, client_order_id="test-003")
    om.register(req)

    fill = Fill(
        client_order_id="test-003",
        exchange_order_id="ex-1",
        symbol="BTC-USD",
        side="buy",
        price=50000.0,
        size=0.4,
        fee=0.01,
        trade_id="t1",
    )
    result = om.apply_fill(fill)
    assert result.status == "partially_filled"
    assert result.filled_size == 0.4
    assert result.avg_fill_price == 50000.0


def test_apply_fill_complete():
    om = OrderManager()
    req = OrderRequest(symbol="BTC-USD", side="buy", size=1.0, client_order_id="test-004")
    om.register(req)

    # First fill
    om.apply_fill(Fill(
        client_order_id="test-004", exchange_order_id="ex-1",
        symbol="BTC-USD", side="buy", price=50000.0, size=0.6,
        fee=0.01, trade_id="t1",
    ))

    # Second fill completes the order
    result = om.apply_fill(Fill(
        client_order_id="test-004", exchange_order_id="ex-1",
        symbol="BTC-USD", side="buy", price=50100.0, size=0.4,
        fee=0.01, trade_id="t2",
    ))
    assert result.status == "filled"
    assert result.filled_size == 1.0
    # Weighted avg: (50000*0.6 + 50100*0.4) / 1.0 = 50040.0
    assert abs(result.avg_fill_price - 50040.0) < 0.01


def test_cancel_and_reject():
    om = OrderManager()
    req = OrderRequest(symbol="BTC-USD", side="buy", size=1.0, client_order_id="test-005")
    om.register(req)

    om.mark_cancelled("test-005")
    assert om.get("test-005").status == "cancelled"

    req2 = OrderRequest(symbol="ETH-USD", side="sell", size=2.0, client_order_id="test-006")
    om.register(req2)
    om.mark_rejected("test-006", "insufficient funds")
    assert om.get("test-006").status == "rejected"
    assert om.get("test-006").reject_reason == "insufficient funds"


def test_open_orders():
    om = OrderManager()
    for i in range(5):
        req = OrderRequest(symbol="BTC-USD", side="buy", size=0.1, client_order_id=f"o-{i}")
        om.register(req)

    assert len(om.open_orders) == 5
    om.mark_cancelled("o-0")
    om.mark_rejected("o-1", "bad")
    assert len(om.open_orders) == 3


def test_get_by_exchange_id():
    om = OrderManager()
    req = OrderRequest(symbol="BTC-USD", side="buy", size=0.1, client_order_id="test-007")
    om.register(req)
    om.update_exchange_id("test-007", "exchange-xyz")

    found = om.get_by_exchange_id("exchange-xyz")
    assert found is not None
    assert found.client_order_id == "test-007"
    assert om.get_by_exchange_id("nonexistent") is None


def test_order_request_serialization():
    req = OrderRequest(
        symbol="BTC-USD", side="buy", size=0.5,
        order_type="limit", limit_price=45000.0,
        client_order_id="ser-001", time_in_force="IOC",
    )
    d = req.to_dict()
    assert d["symbol"] == "BTC-USD"
    assert d["limit_price"] == 45000.0

    req2 = OrderRequest.from_dict(d)
    assert req2.symbol == req.symbol
    assert req2.time_in_force == "IOC"


def test_fill_serialization():
    fill = Fill(
        client_order_id="f-001", exchange_order_id="ex-1",
        symbol="ETH-USD", side="sell", price=3000.0,
        size=2.0, fee=0.5, trade_id="t-100",
    )
    d = fill.to_dict()
    fill2 = Fill.from_dict(d)
    assert fill2.price == 3000.0
    assert fill2.trade_id == "t-100"
