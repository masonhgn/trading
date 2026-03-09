"""Pre-trade risk checks.

Position limits, loss limits, buying power validation.
Stateless — receives current state, returns decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PositionState:
    """Current position and P&L snapshot."""
    position: int
    max_position: int
    total_pnl_cents: float
    max_loss_cents: float


@dataclass(slots=True)
class RiskCheck:
    allowed: bool
    reason: str | None = None


@dataclass(slots=True)
class SideRiskCheck:
    bid_allowed: bool
    ask_allowed: bool
    bid_reason: str | None = None
    ask_reason: str | None = None


@dataclass(slots=True)
class BuyingPowerCheck:
    allowed: bool
    adjusted_size: int
    reason: str | None = None


class RiskManager:
    """Pre-trade risk gate.

    Pure decision-maker: given state and proposed action, returns
    whether to proceed.  Owns no mutable state.
    """

    def __init__(self, max_position: int, max_loss_cents: float) -> None:
        self.max_position = max_position
        self.max_loss_cents = max_loss_cents

    def check_risk_limits(self, position: PositionState) -> RiskCheck:
        if position.total_pnl_cents < -self.max_loss_cents:
            return RiskCheck(
                allowed=False,
                reason=f"loss_limit_breached:{position.total_pnl_cents:.0f}c",
            )
        return RiskCheck(allowed=True)

    def check_sides(self, position: PositionState) -> SideRiskCheck:
        bid_allowed = position.position < self.max_position
        ask_allowed = position.position > -self.max_position
        return SideRiskCheck(
            bid_allowed=bid_allowed,
            ask_allowed=ask_allowed,
            bid_reason=f"at_max_long:{position.position}" if not bid_allowed else None,
            ask_reason=f"at_max_short:{position.position}" if not ask_allowed else None,
        )

    def check_buying_power(
        self,
        side: str,
        price: float,
        desired_size: int,
        buying_power_cents: float | None,
    ) -> BuyingPowerCheck:
        if buying_power_cents is None:
            return BuyingPowerCheck(allowed=True, adjusted_size=desired_size)

        cost_per = price if side == "bid" else (100 - price)
        if cost_per <= 0:
            return BuyingPowerCheck(allowed=False, adjusted_size=0,
                                   reason=f"invalid_cost:{cost_per:.1f}c")

        max_afford = math.floor(buying_power_cents / cost_per)
        adjusted = min(desired_size, max_afford)

        if adjusted < 1:
            return BuyingPowerCheck(
                allowed=False, adjusted_size=0,
                reason=f"insufficient_bp:{buying_power_cents:.0f}c",
            )

        reason = None
        if adjusted < desired_size:
            reason = f"size_reduced:{desired_size}->{adjusted}"

        return BuyingPowerCheck(allowed=True, adjusted_size=adjusted, reason=reason)
