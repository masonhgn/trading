"""Local order state tracker.

Maintains a map of all orders by client_order_id. Provides fast lookup
and state transitions as fills and status updates arrive.
"""

from __future__ import annotations

import logging
import uuid
from common.models import OrderRequest, OrderStatus, Fill

logger = logging.getLogger(__name__)


def generate_client_order_id() -> str:
    """Generate a unique client order ID."""
    return uuid.uuid4().hex[:16]


class OrderManager:
    """Tracks order lifecycle from request to fill/cancel."""

    def __init__(self) -> None:
        self._orders: dict[str, OrderStatus] = {}  # client_order_id -> status

    def register(self, request: OrderRequest, exchange_order_id: str = "") -> OrderStatus:
        """Register a new order from a request. Returns the initial OrderStatus."""
        coid = request.client_order_id or generate_client_order_id()
        status = OrderStatus(
            client_order_id=coid,
            exchange_order_id=exchange_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            limit_price=request.limit_price,
            size=request.size,
            filled_size=0.0,
            avg_fill_price=0.0,
            status="pending",
        )
        self._orders[coid] = status
        logger.info("Registered order %s: %s %s %s @ %s",
                     coid, request.side, request.size, request.symbol, request.limit_price)
        return status

    def update_exchange_id(self, client_order_id: str, exchange_order_id: str) -> OrderStatus | None:
        """Set the exchange order ID once the exchange acknowledges the order."""
        order = self._orders.get(client_order_id)
        if order:
            order.exchange_order_id = exchange_order_id
            if order.status == "pending":
                order.status = "open"
        return order

    def apply_fill(self, fill: Fill) -> OrderStatus | None:
        """Apply a fill to update order state."""
        order = self._orders.get(fill.client_order_id)
        if not order:
            logger.warning("Fill for unknown order %s", fill.client_order_id)
            return None

        # Update average fill price (weighted average)
        prev_value = order.avg_fill_price * order.filled_size
        new_value = fill.price * fill.size
        order.filled_size += fill.size
        order.avg_fill_price = (prev_value + new_value) / order.filled_size

        # Update status
        if order.filled_size >= order.size:
            order.status = "filled"
        else:
            order.status = "partially_filled"

        logger.info("Fill on %s: %.4f @ %.2f (total filled: %.4f / %.4f)",
                     fill.client_order_id, fill.size, fill.price,
                     order.filled_size, order.size)
        return order

    def mark_cancelled(self, client_order_id: str) -> OrderStatus | None:
        """Mark an order as cancelled."""
        order = self._orders.get(client_order_id)
        if order:
            order.status = "cancelled"
        return order

    def mark_rejected(self, client_order_id: str, reason: str = "") -> OrderStatus | None:
        """Mark an order as rejected."""
        order = self._orders.get(client_order_id)
        if order:
            order.status = "rejected"
            order.reject_reason = reason
        return order

    def get(self, client_order_id: str) -> OrderStatus | None:
        return self._orders.get(client_order_id)

    def get_by_exchange_id(self, exchange_order_id: str) -> OrderStatus | None:
        for order in self._orders.values():
            if order.exchange_order_id == exchange_order_id:
                return order
        return None

    @property
    def open_orders(self) -> list[OrderStatus]:
        return [o for o in self._orders.values() if o.status in ("pending", "open", "partially_filled")]

    @property
    def all_orders(self) -> list[OrderStatus]:
        return list(self._orders.values())
