"""Coinbase Advanced Trade data gateway.

Connects via WebSocket, maintains L2 order books, and fires
normalized BookUpdate / TradeUpdate callbacks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable

from coinbase.websocket import WSClient

from gateway.base import BookUpdate, DataGateway, TradeUpdate
from gateway.coinbase.orderbook import OrderBookManager

logger = logging.getLogger(__name__)


def _parse_ts(ts_str: str) -> float:
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


class CoinbaseDataGateway(DataGateway):
    """WebSocket-based market data feed for Coinbase Advanced Trade."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        key_file: str = "",
        channels: list[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._key_file = key_file
        self._channels = channels or ["level2", "market_trades"]
        self._symbols: list[str] = []
        self._books = OrderBookManager()
        self._ws: WSClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._msg_count = 0

        self._book_callbacks: list[Callable[[BookUpdate], None]] = []
        self._trade_callbacks: list[Callable[[TradeUpdate], None]] = []

    @property
    def exchange_name(self) -> str:
        return "coinbase"

    @property
    def books(self) -> OrderBookManager:
        return self._books

    @property
    def msg_count(self) -> int:
        return self._msg_count

    def on_book_update(self, callback: Callable[[BookUpdate], None]) -> None:
        self._book_callbacks.append(callback)

    def on_trade(self, callback: Callable[[TradeUpdate], None]) -> None:
        self._trade_callbacks.append(callback)

    # -- Connection lifecycle ------------------------------------------------

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        ws_kwargs: dict = {"on_message": self._on_message}
        if self._key_file:
            ws_kwargs["key_file"] = self._key_file
        elif self._api_key:
            ws_kwargs["api_key"] = self._api_key
            ws_kwargs["api_secret"] = self._api_secret

        self._ws = WSClient(**ws_kwargs)
        self._ws.open()
        self._running = True
        logger.info("Coinbase WS connected")

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("Coinbase WS disconnected (%d msgs)", self._msg_count)

    async def subscribe(self, symbols: list[str]) -> None:
        self._symbols = symbols
        for channel in self._channels:
            self._ws.subscribe(product_ids=symbols, channels=[channel])
            logger.info("Subscribed to %s for %s", channel, symbols)

    async def run(self) -> None:
        await self.connect()
        await self.subscribe(self._symbols)
        try:
            while self._running:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, self._ws.run_forever_with_exception_check
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Coinbase WS error: %s. Reconnecting...", e)
                    await asyncio.sleep(1)
                    await self.connect()
                    await self.subscribe(self._symbols)
        except asyncio.CancelledError:
            pass
        finally:
            await self.disconnect()

    # -- Message handling (runs in WS thread) --------------------------------

    def _on_message(self, msg) -> None:
        import json
        try:
            if isinstance(msg, str):
                data = json.loads(msg)
            elif isinstance(msg, dict):
                data = msg
            elif hasattr(msg, "__dict__"):
                data = msg.__dict__
            else:
                data = msg

            channel = data.get("channel", "")
            events = data.get("events", [])

            for event in events:
                self._msg_count += 1
                if channel == "l2_data":
                    self._handle_l2(event, data)
                elif channel == "market_trades":
                    self._handle_trades(event)
        except Exception:
            logger.exception("Error processing Coinbase message")

    def _handle_l2(self, event: dict, raw: dict) -> None:
        ts = _parse_ts(raw.get("timestamp", ""))

        for update in event.get("updates", []):
            symbol = update.get("product_id", event.get("product_id", ""))
            book = self._books.get_or_create(symbol)

            side = update.get("side", "").lower()
            if side in ("offer", "ask", "sell"):
                side = "ask"
            else:
                side = "bid"

            price = update.get("price_level", update.get("price", "0"))
            size = update.get("new_quantity", update.get("quantity", "0"))
            book.apply_update(side, price, size)

        # Fire callbacks for affected symbols
        seen: set[str] = set()
        for update in event.get("updates", []):
            sym = update.get("product_id", event.get("product_id", ""))
            if sym and sym not in seen:
                seen.add(sym)
                ob = self._books.get(sym)
                if ob:
                    bb = ob.best_bid()
                    ba = ob.best_ask()
                    if bb and ba:
                        bu = BookUpdate(
                            exchange="coinbase",
                            symbol=sym,
                            bid=bb[0], ask=ba[0],
                            bid_size=bb[1], ask_size=ba[1],
                            mid=(bb[0] + ba[0]) / 2,
                            ts=ts,
                        )
                        self._fire_book(bu)

    def _handle_trades(self, event: dict) -> None:
        for t in event.get("trades", []):
            tu = TradeUpdate(
                exchange="coinbase",
                symbol=t.get("product_id", ""),
                price=float(t.get("price", 0)),
                size=float(t.get("size", 0)),
                side=t.get("side", "unknown").lower(),
                trade_id=str(t.get("trade_id", "")),
                ts=_parse_ts(t.get("time", "")),
            )
            self._fire_trade(tu)

    def _fire_book(self, update: BookUpdate) -> None:
        for cb in self._book_callbacks:
            self._loop.call_soon_threadsafe(cb, update)

    def _fire_trade(self, update: TradeUpdate) -> None:
        for cb in self._trade_callbacks:
            self._loop.call_soon_threadsafe(cb, update)
