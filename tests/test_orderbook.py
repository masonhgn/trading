"""Tests for the L2 order book."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_service.orderbook import L2OrderBook, OrderBookManager


def test_apply_snapshot():
    book = L2OrderBook("BTC-USD")
    book.apply_snapshot(
        bids=[["50000", "1.0"], ["49999", "2.0"]],
        asks=[["50001", "0.5"], ["50002", "1.5"]],
    )
    assert book.best_bid == 50000.0
    assert book.best_ask == 50001.0


def test_apply_update():
    book = L2OrderBook("BTC-USD")
    book.apply_snapshot(
        bids=[["100", "1.0"]],
        asks=[["101", "1.0"]],
    )

    # Update existing level
    book.apply_update("bid", "100", "2.0")
    snap = book.snapshot()
    assert snap.bids[0].size == 2.0

    # Add new level
    book.apply_update("bid", "99", "3.0")
    snap = book.snapshot()
    assert len(snap.bids) == 2
    assert snap.bids[0].price == 100.0  # still top
    assert snap.bids[1].price == 99.0

    # Remove level (size=0)
    book.apply_update("bid", "100", "0")
    snap = book.snapshot()
    assert len(snap.bids) == 1
    assert snap.best_bid == 99.0


def test_snapshot_depth():
    book = L2OrderBook("ETH-USD")
    for i in range(50):
        book.apply_update("bid", str(3000 - i), "1.0")
        book.apply_update("ask", str(3001 + i), "1.0")

    snap = book.snapshot(depth=10)
    assert len(snap.bids) == 10
    assert len(snap.asks) == 10
    assert snap.bids[0].price == 3000.0  # highest bid
    assert snap.asks[0].price == 3001.0  # lowest ask


def test_snapshot_properties():
    book = L2OrderBook("BTC-USD")
    book.apply_snapshot(
        bids=[["50000", "1.0"]],
        asks=[["50010", "1.0"]],
    )
    snap = book.snapshot()
    assert snap.mid == 50005.0
    assert snap.spread == 10.0


def test_manager():
    mgr = OrderBookManager()
    book = mgr.get_or_create("BTC-USD")
    assert book.symbol == "BTC-USD"
    assert mgr.get("BTC-USD") is book
    assert mgr.get("DOGE-USD") is None
    assert "BTC-USD" in mgr.symbols


def test_snapshot_serialization():
    book = L2OrderBook("BTC-USD")
    book.apply_snapshot(
        bids=[["50000", "1.0"], ["49999", "2.0"]],
        asks=[["50001", "0.5"]],
    )
    snap = book.snapshot()
    d = snap.to_dict()

    assert d["symbol"] == "BTC-USD"
    assert d["bids"][0] == [50000.0, 1.0]
    assert d["asks"][0] == [50001.0, 0.5]

    # Round-trip
    from common.models import OrderBookSnapshot
    snap2 = OrderBookSnapshot.from_dict(d)
    assert snap2.best_bid == 50000.0
    assert snap2.best_ask == 50001.0
