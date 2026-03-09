"""Tests for PositionManager."""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from risk.position_manager import PositionManager
from risk.risk_manager import PositionState

# Import helpers — conftest.py is in the same directory
sys.path.insert(0, os.path.dirname(__file__))
from conftest import make_fill, make_book_update


# ---------------------------------------------------------------------------
# Basic state
# ---------------------------------------------------------------------------

def test_initial_state():
    pm = PositionManager("TEST")
    assert pm.position == 0
    assert pm.cash_cents == 0.0
    assert pm.avg_entry_price == 0.0
    assert pm.entry_side == ""


def test_initialize_sets_position():
    pm = PositionManager("TEST")
    pm.initialize(position=5, cash_cents=100.0)
    assert pm.position == 5
    assert pm.cash_cents == 100.0


# ---------------------------------------------------------------------------
# Fill processing
# ---------------------------------------------------------------------------

def test_process_buy_fill_increments_position():
    pm = PositionManager("TEST")
    fill = make_fill(action="buy", side="yes", price=40.0, size=3, fee=0.0)
    pm.process_fill(fill)
    assert pm.position == 3
    # cash = -(price * size + fee) = -(40 * 3 + 0) = -120
    assert pm.cash_cents == -120.0


def test_process_sell_fill_decrements_position():
    pm = PositionManager("TEST")
    # Start with a position so sell doesn't clamp to 0
    pm.initialize(position=5, cash_cents=-200.0)
    fill = make_fill(action="sell", side="yes", price=60.0, size=2, fee=0.0)
    pm.process_fill(fill)
    assert pm.position == 3
    # cash = -200 + (price * size - fee) = -200 + 120 = -80
    assert pm.cash_cents == -80.0


# ---------------------------------------------------------------------------
# Position clamping
# ---------------------------------------------------------------------------

def test_position_clamped_to_zero():
    """Selling more than we hold clamps position to 0, not negative."""
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", price=50.0, size=2))
    pm.process_fill(make_fill(action="sell", price=50.0, size=5))
    assert pm.position == 0


def test_position_clamped_exact():
    """Selling exactly what we hold results in 0."""
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", price=50.0, size=2))
    pm.process_fill(make_fill(action="sell", price=50.0, size=2))
    assert pm.position == 0
    assert pm.avg_entry_price == 0.0
    assert pm.entry_side == ""


# ---------------------------------------------------------------------------
# Average entry price
# ---------------------------------------------------------------------------

def test_avg_entry_price_single_fill():
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", side="yes", price=45.0, size=1))
    assert pm.avg_entry_price == 45.0
    assert pm.entry_side == "yes"


def test_avg_entry_price_multiple_fills_same_side():
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", side="yes", price=40.0, size=2))
    pm.process_fill(make_fill(action="buy", side="yes", price=60.0, size=2))
    # avg = (40*2 + 60*2) / 4 = 200 / 4 = 50
    assert pm.avg_entry_price == pytest.approx(50.0)
    assert pm.entry_side == "yes"


def test_avg_entry_price_reset_on_side_switch():
    """When entry side switches, old average is discarded."""
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", side="yes", price=40.0, size=2))
    assert pm.avg_entry_price == 40.0
    assert pm.entry_side == "yes"

    # Switch to "no" side — average should reset to the new fill's price
    pm.process_fill(make_fill(action="buy", side="no", price=70.0, size=1))
    assert pm.avg_entry_price == 70.0
    assert pm.entry_side == "no"


def test_avg_entry_price_reset_on_flat():
    """Going flat (sell all) resets avg_entry and entry_side."""
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", side="yes", price=45.0, size=3))
    pm.process_fill(make_fill(action="sell", side="yes", price=55.0, size=3))
    assert pm.position == 0
    assert pm.avg_entry_price == 0.0
    assert pm.entry_side == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_reject_zero_price_fill():
    """A fill with price <= 0 is rejected (no state change)."""
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", price=0.0, size=1))
    assert pm.position == 0
    assert pm.cash_cents == 0.0

    pm.process_fill(make_fill(action="buy", price=-5.0, size=1))
    assert pm.position == 0


# ---------------------------------------------------------------------------
# Fee and counter tracking
# ---------------------------------------------------------------------------

def test_fee_tracking():
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", price=50.0, size=1, fee=1.5))
    pm.process_fill(make_fill(action="sell", price=55.0, size=1, fee=2.0))
    assert pm._total_fees == pytest.approx(3.5)
    # cash from buy: -(50*1 + 1.5) = -51.5
    # cash from sell: +(55*1 - 2.0) = +53.0   (position goes to 0, clamped)
    assert pm.cash_cents == pytest.approx(1.5)


def test_fill_counters():
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", price=50.0, size=1))
    pm.process_fill(make_fill(action="buy", price=51.0, size=1))
    pm.process_fill(make_fill(action="sell", price=55.0, size=1))

    assert pm._fill_count == 3
    assert pm._buy_fills == 2
    assert pm._sell_fills == 1


# ---------------------------------------------------------------------------
# Position reconciliation
# ---------------------------------------------------------------------------

def test_process_position_update_corrects_position():
    pm = PositionManager("TEST")
    pm.initialize(position=3)
    drift = pm.process_position_update(
        exchange_position=5,
        realized_pnl_cents=100.0,
        fees_paid_cents=2.0,
    )
    assert pm.position == 5
    assert drift == 2  # 5 - 3


def test_process_position_update_logs_drift(caplog):
    pm = PositionManager("TEST")
    pm._cash_cents = 100.0
    pm._total_fees = 10.0

    with caplog.at_level(logging.WARNING, logger="risk.position_manager"):
        pm.process_position_update(
            exchange_position=0,
            realized_pnl_cents=0.0,    # cash drift = |100 - 0| = 100 > 5
            fees_paid_cents=10.0,      # fee drift = 0 (no log)
        )
    assert any("Cash drift" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# P&L helpers
# ---------------------------------------------------------------------------

def test_get_pnl_with_mid():
    pm = PositionManager("TEST")
    pm.initialize(position=0)
    pm.process_fill(make_fill(action="buy", price=40.0, size=5, fee=1.0))

    pnl = pm.get_pnl(mid=50.0)

    assert pnl["position"] == 5
    # cash = -(40*5 + 1) = -201
    assert pnl["cash_cents"] == pytest.approx(-201.0)
    # mtm = 5 * 50 = 250
    assert pnl["mark_to_market"] == pytest.approx(250.0)
    # total_pnl = -201 + 250 = 49
    assert pnl["total_pnl_cents"] == pytest.approx(49.0)
    assert pnl["fill_count"] == 1
    assert pnl["buy_fills"] == 1
    assert pnl["sell_fills"] == 0
    assert pnl["total_fees_paid"] == pytest.approx(1.0)


def test_get_pnl_no_mid():
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", price=40.0, size=2))

    pnl = pm.get_pnl(mid=None)

    assert pnl["mark_to_market"] == 0
    assert pnl["total_pnl_cents"] == pnl["cash_cents"]


def test_get_position_state():
    pm = PositionManager("TEST")
    pm.process_fill(make_fill(action="buy", price=40.0, size=3, fee=0.5))

    state = pm.get_position_state(mid=50.0, max_position=10, max_loss_cents=500.0)

    assert isinstance(state, PositionState)
    assert state.position == 3
    assert state.max_position == 10
    # cash = -(40*3 + 0.5) = -120.5; mtm = 3*50 = 150; total = 29.5
    assert state.total_pnl_cents == pytest.approx(29.5)
    assert state.max_loss_cents == 500.0


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

import pytest  # noqa: E402 — late import is fine, pytest is already loaded
