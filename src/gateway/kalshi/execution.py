"""Kalshi order gateway.

Wraps the KalshiRestClient behind the OrderGateway interface.
All blocking REST calls run in an executor.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Callable

from gateway.base import (
    FillUpdate,
    OrderGateway,
    OrderRequest,
    OrderResponse,
    PositionInfo,
)
from gateway.kalshi.client import KalshiRestClient

logger = logging.getLogger(__name__)


class KalshiOrderGateway(OrderGateway):
    """Order entry via Kalshi REST API."""

    def __init__(self, client: KalshiRestClient) -> None:
        self._client = client
        self._fill_callbacks: list[Callable[[FillUpdate], None]] = []

    @property
    def exchange_name(self) -> str:
        return "kalshi"

    async def connect(self) -> None:
        # REST client is stateless — just verify auth works
        loop = asyncio.get_running_loop()
        try:
            bal = await loop.run_in_executor(None, self._client.get_balance)
            cents = bal.get("balance", 0)
            logger.info("Kalshi connected, balance: $%.2f", cents / 100)
        except Exception as e:
            logger.error("Kalshi auth check failed: %s", e)
            raise

    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        """Submit an order to Kalshi.

        req.side should be "yes" or "no".
        req.action should be "buy" or "sell".
        req.price is in cents (1-99).
        """
        loop = asyncio.get_running_loop()
        client_oid = req.client_order_id or f"ka-{int(time.time()*1000)}"

        # Build Kalshi-specific params
        kwargs: dict = {
            "ticker": req.symbol,
            "client_order_id": client_oid,
            "side": req.side,
            "action": req.action,
            "count": int(req.size),
            "order_type": req.order_type,
        }
        if req.price is not None:
            # Kalshi accepts yes_price for both sides
            if req.side == "yes":
                kwargs["yes_price"] = round(req.price)
            else:
                kwargs["no_price"] = round(req.price)

        try:
            resp = await loop.run_in_executor(
                None,
                functools.partial(self._client.create_order, **kwargs),
            )
            order_id = resp.get("order", {}).get("order_id", "")
            if order_id:
                return OrderResponse(success=True, order_id=order_id)
            return OrderResponse(success=False, error=str(resp))
        except Exception as e:
            logger.error("Kalshi order failed: %s", e)
            return OrderResponse(success=False, error=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                functools.partial(self._client.cancel_order, order_id),
            )
            return True
        except Exception as e:
            if "404" in str(e):
                return True  # already filled or cancelled
            logger.error("Kalshi cancel failed: %s", e)
            return False

    async def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by ticker."""
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                functools.partial(
                    self._client.get_orders,
                    ticker=symbol,
                    status="resting",
                ),
            )
            orders = resp.get("orders", [])
            cancelled = 0
            for order in orders:
                oid = order.get("order_id")
                if oid and await self.cancel_order(oid):
                    cancelled += 1
            return cancelled
        except Exception as e:
            logger.error("Kalshi cancel_all failed: %s", e)
            return 0

    async def get_positions(self, symbol: str | None = None) -> list[PositionInfo]:
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                functools.partial(self._client.get_positions, ticker=symbol),
            )
            positions = []
            for item in resp.get("market_positions", []):
                positions.append(PositionInfo(
                    symbol=item.get("market_ticker", ""),
                    position=int(item.get("position", 0)),
                    realized_pnl=float(item.get("realized_pnl", 0)),  # cents
                    fees_paid=float(item.get("fees_paid", 0)),  # cents
                ))
            return positions
        except Exception as e:
            logger.error("Kalshi get_positions failed: %s", e)
            return []

    async def get_balance(self) -> float:
        """Get available balance in cents."""
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, self._client.get_balance)
            return float(resp.get("balance", 0))
        except Exception as e:
            logger.error("Kalshi balance query failed: %s", e)
            return 0.0

    def on_fill(self, callback: Callable[[FillUpdate], None]) -> None:
        self._fill_callbacks.append(callback)
