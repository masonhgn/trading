"""Coinbase Advanced Trade order gateway.

Wraps the coinbase-advanced-py REST client behind the OrderGateway
interface.  All blocking REST calls run in an executor.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Callable

from coinbase.rest import RESTClient

from gateway.base import (
    FillUpdate,
    OrderGateway,
    OrderRequest,
    OrderResponse,
    PositionInfo,
)

logger = logging.getLogger(__name__)


class CoinbaseOrderGateway(OrderGateway):
    """Order entry via Coinbase Advanced Trade REST API."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        key_file: str = "",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._key_file = key_file
        self._client: RESTClient | None = None

    @property
    def exchange_name(self) -> str:
        return "coinbase"

    async def connect(self) -> None:
        if self._key_file:
            self._client = RESTClient(key_file=self._key_file)
        else:
            self._client = RESTClient(
                api_key=self._api_key, api_secret=self._api_secret
            )
        logger.info("Coinbase REST client initialized")

    async def submit_order(self, req: OrderRequest) -> OrderResponse:
        loop = asyncio.get_running_loop()
        client_oid = req.client_order_id or f"cb-{int(time.time()*1000)}"

        try:
            if req.order_type == "market":
                resp = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self._client.market_order,
                        client_order_id=client_oid,
                        product_id=req.symbol,
                        side=req.side.upper(),
                        quote_size=str(req.size) if req.side == "buy" else None,
                        base_size=str(req.size) if req.side == "sell" else None,
                    ),
                )
            else:
                if req.price is None:
                    return OrderResponse(success=False, error="limit_price required")
                resp = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self._client.limit_order_gtc,
                        client_order_id=client_oid,
                        product_id=req.symbol,
                        side=req.side.upper(),
                        base_size=str(req.size),
                        limit_price=str(req.price),
                    ),
                )

            oid = resp.get("order_id", resp.get("success_response", {}).get("order_id", ""))
            if oid:
                return OrderResponse(success=True, order_id=oid)
            return OrderResponse(success=False, error=str(resp))

        except Exception as e:
            logger.error("Coinbase order failed: %s", e)
            return OrderResponse(success=False, error=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                functools.partial(self._client.cancel_orders, order_ids=[order_id]),
            )
            return True
        except Exception as e:
            logger.error("Coinbase cancel failed: %s", e)
            return False

    async def cancel_all(self, symbol: str | None = None) -> int:
        # Coinbase doesn't have a "cancel all for symbol" — skip for now
        return 0

    async def get_positions(self, symbol: str | None = None) -> list[PositionInfo]:
        # Coinbase spot doesn't have "positions" in the futures sense
        return []

    async def get_balance(self) -> float:
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, self._client.get_accounts)
            accounts = resp.get("accounts", [])
            for acct in accounts:
                if acct.get("currency") == "USD":
                    return float(acct.get("available_balance", {}).get("value", 0))
            return 0.0
        except Exception as e:
            logger.error("Coinbase balance query failed: %s", e)
            return 0.0
