"""Position and P&L tracking.

Single source of truth for position, cash, unrealized/realized P&L.
Exchange-agnostic — works with any gateway that produces FillUpdates.
"""

from __future__ import annotations

import logging
from typing import Optional

from gateway.base import FillUpdate, PositionUpdate
from risk.risk_manager import PositionState

logger = logging.getLogger(__name__)


class PositionManager:
    """Tracks position, cash, and P&L for a single instrument."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

        self._position: int = 0
        self._cash_cents: float = 0.0

        self._total_fees: float = 0.0
        self._fill_count: int = 0
        self._buy_fills: int = 0
        self._sell_fills: int = 0

        # Entry tracking for take-profit / stop-loss
        self._avg_entry_price: float = 0.0  # cents
        self._entry_side: str = ""  # "yes" or "no"

        # Last reconciled values from exchange
        self._exchange_realized_pnl: float | None = None
        self._exchange_fees: float | None = None

    @property
    def position(self) -> int:
        return self._position

    @property
    def cash_cents(self) -> float:
        return self._cash_cents

    @property
    def avg_entry_price(self) -> float:
        return self._avg_entry_price

    @property
    def entry_side(self) -> str:
        return self._entry_side

    def initialize(self, position: int, cash_cents: float = 0.0) -> None:
        """Set initial position from exchange on startup."""
        self._position = position
        self._cash_cents = cash_cents
        logger.info("Position initialized: %s pos=%d cash=%.2f",
                     self.symbol, position, cash_cents)

    def process_fill(self, fill: FillUpdate) -> None:
        """Update cash tracking from a fill."""
        if fill.price <= 0:
            logger.error("Fill with zero price: %s", fill)
            return

        if fill.action == "buy":
            self._position += int(fill.size)
            self._cash_cents -= (fill.price * fill.size + fill.fee)
            # Update average entry price
            old_qty = abs(self._position - int(fill.size))
            if self._entry_side and self._entry_side != fill.side:
                old_qty = 0  # switching sides, reset
            new_qty = old_qty + fill.size
            if new_qty > 0:
                self._avg_entry_price = (
                    (self._avg_entry_price * old_qty + fill.price * fill.size)
                    / new_qty
                )
            self._entry_side = fill.side
        else:
            self._position -= int(fill.size)
            self._cash_cents += (fill.price * fill.size - fill.fee)
            # Clamp position to 0 — don't allow negative
            if self._position <= 0:
                if self._position < 0:
                    logger.warning("Position went negative (%d), clamping to 0", self._position)
                self._position = 0
                self._avg_entry_price = 0.0
                self._entry_side = ""

        self._fill_count += 1
        self._total_fees += fill.fee
        if fill.action == "buy":
            self._buy_fills += 1
        else:
            self._sell_fills += 1

    def process_position_update(
        self,
        exchange_position: int,
        realized_pnl_cents: float,
        fees_paid_cents: float,
    ) -> int:
        """Handle authoritative position update from exchange.

        Returns position change (new - old).
        """
        old = self._position
        self._position = exchange_position
        self._exchange_realized_pnl = realized_pnl_cents
        self._exchange_fees = fees_paid_cents

        # Log drift
        fee_drift = abs(self._total_fees - fees_paid_cents)
        if fee_drift > 1.0:
            logger.info("Fee drift: ours=%.2f exchange=%.2f",
                        self._total_fees, fees_paid_cents)

        cash_drift = abs(self._cash_cents - realized_pnl_cents)
        if cash_drift > 5.0:
            logger.warning("Cash drift: ours=%.2f exchange=%.2f",
                           self._cash_cents, realized_pnl_cents)

        return exchange_position - old

    def process_ws_position(self, update: PositionUpdate) -> int:
        """Handle real-time position update from exchange WebSocket.

        This is the authoritative source of truth — replaces REST polling.
        Returns position change (new - old).
        """
        old = self._position

        if old != update.position:
            logger.info("Position update (WS): %s %d → %d",
                        self.symbol, old, update.position)

        self._position = update.position
        self._exchange_realized_pnl = update.realized_pnl
        self._exchange_fees = update.fees_paid

        # Reset entry tracking if position went flat
        if update.position == 0 and old != 0:
            self._avg_entry_price = 0.0
            self._entry_side = ""

        return update.position - old

    def get_pnl(self, mid: float | None) -> dict:
        if mid is not None and self._position > 0:
            # For NO positions, our value is 100 - yes_mid per contract
            if self._entry_side == "no":
                mtm = self._position * (100 - mid)
            else:
                mtm = self._position * mid
        else:
            mtm = 0
        return {
            "position": self._position,
            "cash_cents": self._cash_cents,
            "mark_to_market": mtm,
            "total_pnl_cents": self._cash_cents + mtm,
            "fill_count": self._fill_count,
            "buy_fills": self._buy_fills,
            "sell_fills": self._sell_fills,
            "total_fees_paid": self._total_fees,
        }

    def get_position_state(
        self,
        mid: float | None,
        max_position: int,
        max_loss_cents: float,
    ) -> PositionState:
        if mid is not None and self._position > 0 and self._entry_side == "no":
            mtm = self._position * (100 - mid)
        else:
            mtm = self._position * mid if mid else 0
        return PositionState(
            position=self._position,
            max_position=max_position,
            total_pnl_cents=self._cash_cents + mtm,
            max_loss_cents=max_loss_cents,
        )
