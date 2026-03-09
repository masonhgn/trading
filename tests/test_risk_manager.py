"""Tests for RiskManager pre-trade risk checks."""

import math

import pytest

from risk.risk_manager import (
    BuyingPowerCheck,
    PositionState,
    RiskCheck,
    RiskManager,
    SideRiskCheck,
)


@pytest.fixture
def rm() -> RiskManager:
    """RiskManager with max_position=5, max_loss=500 cents."""
    return RiskManager(max_position=5, max_loss_cents=500.0)


def _pos(position: int = 0, pnl: float = 0.0) -> PositionState:
    """Shortcut to build a PositionState for testing."""
    return PositionState(
        position=position,
        max_position=5,
        total_pnl_cents=pnl,
        max_loss_cents=500.0,
    )


# ── check_risk_limits ────────────────────────────────────────────────


class TestCheckRiskLimits:
    def test_loss_limit_not_breached(self, rm: RiskManager):
        result = rm.check_risk_limits(_pos(pnl=-200.0))
        assert result.allowed is True
        assert result.reason is None

    def test_loss_limit_breached(self, rm: RiskManager):
        result = rm.check_risk_limits(_pos(pnl=-600.0))
        assert result.allowed is False
        assert "loss_limit_breached" in result.reason

    def test_loss_limit_at_boundary(self, rm: RiskManager):
        """Exactly at -max_loss_cents should still be allowed (not strictly less)."""
        result = rm.check_risk_limits(_pos(pnl=-500.0))
        assert result.allowed is True

    def test_loss_limit_positive_pnl(self, rm: RiskManager):
        result = rm.check_risk_limits(_pos(pnl=100.0))
        assert result.allowed is True


# ── check_sides ──────────────────────────────────────────────────────


class TestCheckSides:
    def test_check_sides_below_max(self, rm: RiskManager):
        result = rm.check_sides(_pos(position=2))
        assert result.bid_allowed is True
        assert result.ask_allowed is True
        assert result.bid_reason is None
        assert result.ask_reason is None

    def test_check_sides_at_max_long(self, rm: RiskManager):
        result = rm.check_sides(_pos(position=5))
        assert result.bid_allowed is False
        assert result.ask_allowed is True
        assert "at_max_long" in result.bid_reason

    def test_check_sides_at_max_short(self, rm: RiskManager):
        result = rm.check_sides(_pos(position=-5))
        assert result.bid_allowed is True
        assert result.ask_allowed is False
        assert "at_max_short" in result.ask_reason

    def test_check_sides_zero_position(self, rm: RiskManager):
        result = rm.check_sides(_pos(position=0))
        assert result.bid_allowed is True
        assert result.ask_allowed is True


# ── check_buying_power ───────────────────────────────────────────────


class TestCheckBuyingPower:
    def test_buying_power_none_skips_check(self, rm: RiskManager):
        """When buying_power_cents is None, always allow full size."""
        result = rm.check_buying_power("bid", 50.0, 10, None)
        assert result.allowed is True
        assert result.adjusted_size == 10
        assert result.reason is None

    def test_buying_power_sufficient(self, rm: RiskManager):
        # bid at 40c, want 5 contracts => need 200c, have 500c
        result = rm.check_buying_power("bid", 40.0, 5, 500.0)
        assert result.allowed is True
        assert result.adjusted_size == 5
        assert result.reason is None

    def test_buying_power_reduces_size(self, rm: RiskManager):
        # bid at 40c, want 10 contracts => need 400c, have 150c => afford 3
        result = rm.check_buying_power("bid", 40.0, 10, 150.0)
        assert result.allowed is True
        assert result.adjusted_size == 3
        assert "size_reduced" in result.reason

    def test_buying_power_insufficient(self, rm: RiskManager):
        # bid at 60c, want 5 => need 300c, have 50c => afford 0
        result = rm.check_buying_power("bid", 60.0, 5, 50.0)
        assert result.allowed is False
        assert result.adjusted_size == 0
        assert "insufficient_bp" in result.reason

    def test_buying_power_ask_side_cost(self, rm: RiskManager):
        # ask at 40c => cost is 100-40=60c per contract
        # want 3 => need 180c, have 200c => afford 3
        result = rm.check_buying_power("ask", 40.0, 3, 200.0)
        assert result.allowed is True
        assert result.adjusted_size == 3

    def test_buying_power_invalid_cost(self, rm: RiskManager):
        # bid at 0c => cost_per = 0 => invalid
        result = rm.check_buying_power("bid", 0.0, 5, 500.0)
        assert result.allowed is False
        assert result.adjusted_size == 0
        assert "invalid_cost" in result.reason

    def test_buying_power_ask_at_100(self, rm: RiskManager):
        # ask at 100c => cost = 100-100 = 0 => invalid
        result = rm.check_buying_power("ask", 100.0, 5, 500.0)
        assert result.allowed is False
        assert "invalid_cost" in result.reason
