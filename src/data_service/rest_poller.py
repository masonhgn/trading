"""REST API poller for data that isn't available via WebSocket.

Periodically fetches candles, product info, and order book snapshots
for bootstrapping or gap-filling.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time

from coinbase.rest import RESTClient

from common.msg_bus import Publisher

logger = logging.getLogger(__name__)


class RESTPoller:
    """Polls Coinbase REST API on intervals and publishes to the bus."""

    def __init__(
        self,
        symbols: list[str],
        publisher: Publisher,
        api_key: str = "",
        api_secret: str = "",
        key_file: str = "",
        candle_interval_sec: float = 60.0,
    ) -> None:
        self._symbols = symbols
        self._pub = publisher
        self._candle_interval = candle_interval_sec
        self._running = False

        client_kwargs: dict = {}
        if key_file:
            client_kwargs["key_file"] = key_file
        elif api_key:
            client_kwargs["api_key"] = api_key
            client_kwargs["api_secret"] = api_secret

        self._client = RESTClient(**client_kwargs)

    async def poll_candles(self) -> None:
        """Fetch latest candles for all symbols and publish."""
        loop = asyncio.get_running_loop()
        for symbol in self._symbols:
            try:
                candles = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self._client.get_candles,
                        product_id=symbol,
                        start=str(int(time.time()) - 3600),
                        end=str(int(time.time())),
                        granularity="ONE_MINUTE",
                    ),
                )
                if candles and hasattr(candles, "candles"):
                    candle_list = []
                    for c in candles.candles:
                        candle_list.append({
                            "start": c.get("start") if isinstance(c, dict) else getattr(c, "start", ""),
                            "open": float(c.get("open", 0) if isinstance(c, dict) else getattr(c, "open", 0)),
                            "high": float(c.get("high", 0) if isinstance(c, dict) else getattr(c, "high", 0)),
                            "low": float(c.get("low", 0) if isinstance(c, dict) else getattr(c, "low", 0)),
                            "close": float(c.get("close", 0) if isinstance(c, dict) else getattr(c, "close", 0)),
                            "volume": float(c.get("volume", 0) if isinstance(c, dict) else getattr(c, "volume", 0)),
                        })
                    await self._pub.publish(f"candles.{symbol}", {
                        "symbol": symbol,
                        "granularity": "ONE_MINUTE",
                        "candles": candle_list,
                        "ts": time.time(),
                    })
            except Exception:
                logger.exception("Failed to fetch candles for %s", symbol)

    async def poll_book_snapshot(self, symbol: str) -> dict | None:
        """Fetch a REST order book snapshot for bootstrapping."""
        try:
            loop = asyncio.get_running_loop()
            book = await loop.run_in_executor(
                None,
                functools.partial(self._client.get_product_book, product_id=symbol),
            )
            if book:
                return {
                    "symbol": symbol,
                    "bids": book.get("pricebook", {}).get("bids", []) if isinstance(book, dict) else [],
                    "asks": book.get("pricebook", {}).get("asks", []) if isinstance(book, dict) else [],
                    "ts": time.time(),
                }
        except Exception:
            logger.exception("Failed to fetch book for %s", symbol)
        return None

    async def run(self) -> None:
        """Run the REST poller loop."""
        self._running = True
        logger.info("REST poller started for %s", self._symbols)
        while self._running:
            await self.poll_candles()
            await asyncio.sleep(self._candle_interval)

    def stop(self) -> None:
        self._running = False
