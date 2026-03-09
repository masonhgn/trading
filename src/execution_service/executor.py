"""Order executor — sends orders to Coinbase Advanced Trade API.

Wraps the coinbase-advanced-py RESTClient for order placement,
cancellation, and status queries. All operations update the
local OrderManager and publish state changes to the ZMQ bus.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time

from coinbase.rest import RESTClient

from common.models import OrderRequest, OrderStatus, Fill
from common.msg_bus import Publisher
from execution_service.order_manager import OrderManager, generate_client_order_id

logger = logging.getLogger(__name__)


class Executor:
    """Executes orders against Coinbase and tracks state."""

    def __init__(
        self,
        publisher: Publisher,
        order_manager: OrderManager,
        api_key: str = "",
        api_secret: str = "",
        key_file: str = "",
    ) -> None:
        self._pub = publisher
        self._om = order_manager

        client_kwargs: dict = {}
        if key_file:
            client_kwargs["key_file"] = key_file
        elif api_key:
            client_kwargs["api_key"] = api_key
            client_kwargs["api_secret"] = api_secret
        self._client = RESTClient(**client_kwargs)

    async def submit_order(self, request: OrderRequest) -> OrderStatus:
        """Submit an order to Coinbase. Returns the resulting OrderStatus."""
        if not request.client_order_id:
            request.client_order_id = generate_client_order_id()

        # Register locally first
        status = self._om.register(request)

        try:
            loop = asyncio.get_running_loop()
            if request.order_type == "market":
                resp = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self._client.market_order,
                        client_order_id=request.client_order_id,
                        product_id=request.symbol,
                        side=request.side.upper(),
                        quote_size=str(request.size) if request.side == "buy" else None,
                        base_size=str(request.size) if request.side == "sell" else None,
                    ),
                )
            else:
                if request.limit_price is None:
                    self._om.mark_rejected(request.client_order_id, "limit_price required for limit orders")
                    status = self._om.get(request.client_order_id)
                    await self._pub.publish(f"order.{status.symbol}", status.to_dict())
                    return status
                resp = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self._client.limit_order_gtc,
                        client_order_id=request.client_order_id,
                        product_id=request.symbol,
                        side=request.side.upper(),
                        base_size=str(request.size),
                        limit_price=str(request.limit_price),
                    ),
                )

            # Parse response
            if isinstance(resp, dict):
                success = resp.get("success", False)
                order_data = resp.get("order_configuration", resp)
                exchange_id = resp.get("order_id", "")
                failure_reason = resp.get("failure_reason", "")
            else:
                success = getattr(resp, "success", False)
                exchange_id = getattr(resp, "order_id", "")
                failure_reason = getattr(resp, "failure_reason", "")

            if success or exchange_id:
                self._om.update_exchange_id(request.client_order_id, exchange_id)
                status = self._om.get(request.client_order_id)
                logger.info("Order accepted: %s -> %s", request.client_order_id, exchange_id)
            else:
                self._om.mark_rejected(request.client_order_id, failure_reason)
                status = self._om.get(request.client_order_id)
                logger.warning("Order rejected: %s reason=%s", request.client_order_id, failure_reason)

        except Exception as e:
            self._om.mark_rejected(request.client_order_id, str(e))
            status = self._om.get(request.client_order_id)
            logger.exception("Order submission failed: %s", request.client_order_id)

        # Publish status update
        await self._pub.publish(f"order.{status.symbol}", status.to_dict())
        return status

    async def cancel_order(self, client_order_id: str) -> OrderStatus | None:
        """Cancel an order by client_order_id."""
        order = self._om.get(client_order_id)
        if not order:
            logger.warning("Cancel: unknown order %s", client_order_id)
            return None

        if order.status in ("filled", "cancelled", "rejected"):
            logger.warning("Cancel: order %s already %s", client_order_id, order.status)
            return order

        try:
            # Coinbase cancel takes exchange order IDs
            if order.exchange_order_id:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        self._client.cancel_orders,
                        order_ids=[order.exchange_order_id],
                    ),
                )
            self._om.mark_cancelled(client_order_id)
            logger.info("Cancelled order %s", client_order_id)
        except Exception as e:
            logger.exception("Cancel failed for %s: %s", client_order_id, e)

        order = self._om.get(client_order_id)
        await self._pub.publish(f"order.{order.symbol}", order.to_dict())
        return order

    async def cancel_all(self, symbol: str | None = None) -> list[OrderStatus]:
        """Cancel all open orders, optionally filtered by symbol."""
        cancelled = []
        for order in self._om.open_orders:
            if symbol and order.symbol != symbol:
                continue
            result = await self.cancel_order(order.client_order_id)
            if result:
                cancelled.append(result)
        return cancelled

    async def sync_order(self, client_order_id: str) -> OrderStatus | None:
        """Query Coinbase for current order state and sync locally."""
        order = self._om.get(client_order_id)
        if not order or not order.exchange_order_id:
            return order

        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                functools.partial(self._client.get_order, order_id=order.exchange_order_id),
            )
            if isinstance(resp, dict):
                order_data = resp.get("order", resp)
            else:
                order_data = getattr(resp, "order", resp)

            if isinstance(order_data, dict):
                cb_status = order_data.get("status", "")
                filled = float(order_data.get("filled_size", 0))
                avg_price = float(order_data.get("average_filled_price", 0))
            else:
                cb_status = getattr(order_data, "status", "")
                filled = float(getattr(order_data, "filled_size", 0))
                avg_price = float(getattr(order_data, "average_filled_price", 0))

            order.filled_size = filled
            order.avg_fill_price = avg_price

            status_map = {
                "OPEN": "open",
                "FILLED": "filled",
                "CANCELLED": "cancelled",
                "EXPIRED": "cancelled",
                "FAILED": "rejected",
                "PENDING": "pending",
            }
            order.status = status_map.get(cb_status, order.status)

            await self._pub.publish(f"order.{order.symbol}", order.to_dict())
        except Exception:
            logger.exception("Failed to sync order %s", client_order_id)

        return order

    async def get_balances(self) -> dict[str, dict]:
        """Fetch account balances. Returns {currency: {available, hold}}."""
        try:
            loop = asyncio.get_running_loop()
            accounts = await loop.run_in_executor(None, self._client.get_accounts)
            balances = {}
            account_list = accounts.get("accounts", []) if isinstance(accounts, dict) else getattr(accounts, "accounts", [])
            for acct in account_list:
                if isinstance(acct, dict):
                    currency = acct.get("currency", "")
                    available = float(acct.get("available_balance", {}).get("value", 0))
                    hold = float(acct.get("hold", {}).get("value", 0))
                else:
                    currency = getattr(acct, "currency", "")
                    available = float(getattr(getattr(acct, "available_balance", None), "value", 0))
                    hold = float(getattr(getattr(acct, "hold", None), "value", 0))
                if currency:
                    balances[currency] = {"available": available, "hold": hold}
            return balances
        except Exception:
            logger.exception("Failed to fetch balances")
            return {}
