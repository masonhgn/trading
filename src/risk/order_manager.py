"""Order lifecycle management.

Tracks active orders, handles cancel-replace logic, matches fills.
Exchange-agnostic — uses OrderGateway for execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from gateway.base import FillUpdate, OrderGateway, OrderRequest, OrderResponse, OrderUpdate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActiveOrder:
    """A resting order we're tracking."""
    order_id: str
    side: str           # "bid" or "ask"
    price: float        # cents
    size: int
    placed_at: float    # timestamp


class OrderManager:
    """Manages order placement, cancellation, and active order tracking.

    Maintains at most one bid and one ask.  Uses an OrderGateway
    for actual exchange communication.
    """

    def __init__(
        self,
        gateway: OrderGateway,
        symbol: str,
        default_size: int = 1,
    ) -> None:
        self._gw = gateway
        self._symbol = symbol
        self._default_size = default_size

        self._active: dict[str, ActiveOrder | None] = {"bid": None, "ask": None}
        self._ws_cleared: dict[str, tuple[str, float]] = {}  # order_id -> (side, timestamp)
        self.order_count: int = 0
        self.cancel_count: int = 0

    @property
    def active_bid(self) -> ActiveOrder | None:
        return self._active["bid"]

    @property
    def active_ask(self) -> ActiveOrder | None:
        return self._active["ask"]

    def has_active(self, side: str) -> bool:
        return self._active.get(side) is not None

    async def place_order(
        self,
        side: str,
        price: float,
        size: int | None = None,
    ) -> ActiveOrder | None:
        """Place a limit order on the given side.

        side: "bid" (buy YES) or "ask" (sell YES = buy NO).
        price: in cents for Kalshi (1-99).
        """
        size = size or self._default_size

        # Guard: don't place if we already have an active order on this side
        if self._active.get(side) is not None:
            logger.warning("Already have active %s order on %s, skipping", side, self._symbol)
            return None

        # Build gateway request
        if side == "bid":
            req = OrderRequest(
                symbol=self._symbol,
                side="yes",
                action="buy",
                size=size,
                order_type="limit",
                price=round(price),
            )
        else:
            # "ask" = sell YES = buy NO
            req = OrderRequest(
                symbol=self._symbol,
                side="no",
                action="buy",
                size=size,
                order_type="limit",
                price=round(100 - price),
            )

        if req.price < 1 or req.price > 99:
            logger.warning("Invalid price: %s %dc (original=%dc)", side, req.price, price)
            return None

        resp = await self._gw.submit_order(req)
        if not resp.success:
            logger.error("Order failed: %s %dc - %s", side, price, resp.error)
            return None

        order = ActiveOrder(
            order_id=resp.order_id,
            side=side,
            price=price,
            size=size,
            placed_at=time.time(),
        )
        self._active[side] = order
        self.order_count += 1
        logger.info("Placed %s %dc x%d id=%s", side, price, size, resp.order_id)
        return order

    async def cancel_order(self, side: str) -> bool:
        order = self._active.get(side)
        if order is None:
            return True

        success = await self._gw.cancel_order(order.order_id)
        if success:
            self._active[side] = None
            self.cancel_count += 1
            logger.info("Cancelled %s %dc id=%s", side, order.price, order.order_id)
        return success

    async def cancel_all(self) -> None:
        for side in ["bid", "ask"]:
            if self._active.get(side):
                await self.cancel_order(side)

    async def update_quote(
        self,
        side: str,
        price: float,
        size: int | None = None,
    ) -> ActiveOrder | None:
        """Cancel existing order on this side and place a new one.

        Skips if the active order already matches the desired price.
        Returns None if cancel fails (leaves existing order in place).
        """
        current = self._active.get(side)
        if current and current.price == price:
            return current

        if current:
            cancelled = await self.cancel_order(side)
            if not cancelled:
                logger.warning("Cancel failed for %s %dc id=%s — skipping new order",
                               side, current.price, current.order_id)
                return None

        return await self.place_order(side, price, size)

    def process_fill(self, fill: FillUpdate) -> str | None:
        """Match a fill to an active order and clear it.

        Returns the side ("bid" or "ask") matched, or None.
        """
        for side in ["bid", "ask"]:
            order = self._active.get(side)
            if order and order.order_id == fill.order_id:
                self._active[side] = None
                logger.info("Fill matched %s %dc x%.0f id=%s",
                            side, order.price, fill.size, fill.order_id)
                return side

        # Check if WS already cleared this order (fill-before-order race is resolved)
        ws_entry = self._ws_cleared.pop(fill.order_id, None)
        if ws_entry:
            ws_side, _ = ws_entry
            logger.debug("Fill for WS-cleared order %s %s %dc x%.0f",
                         ws_side, fill.order_id, fill.price, fill.size)
            return ws_side

        logger.info("Fill unmatched: id=%s side=%s %dc x%.0f",
                     fill.order_id, fill.side, fill.price, fill.size)
        return None

    def process_order_update(self, update: OrderUpdate) -> str | None:
        """Handle authoritative order status from exchange WS.

        Clears _active when the exchange confirms an order is
        executed or canceled, fixing the fill-before-order race.
        Returns the side cleared, or None.
        """
        if update.status not in ("executed", "canceled"):
            return None

        for side in ["bid", "ask"]:
            order = self._active.get(side)
            if order and order.order_id == update.order_id:
                self._active[side] = None
                # Remember this order was cleared by WS so process_fill won't log "unmatched"
                self._ws_cleared[update.order_id] = (side, time.time())
                self._evict_ws_cleared()
                logger.info("Order %s %s %dc id=%s (via WS)",
                            update.status, side, order.price, update.order_id)
                return side

        return None

    def _evict_ws_cleared(self) -> None:
        """Remove stale entries from _ws_cleared (older than 60s)."""
        if len(self._ws_cleared) < 20:
            return
        cutoff = time.time() - 60
        stale = [oid for oid, (_, ts) in self._ws_cleared.items() if ts < cutoff]
        for oid in stale:
            del self._ws_cleared[oid]
