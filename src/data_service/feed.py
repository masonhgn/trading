"""WebSocket feed handler for Coinbase Advanced Trade API.

Connects to the Coinbase WebSocket, subscribes to channels,
normalizes messages, and publishes them on the ZMQ message bus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from coinbase.websocket import WSClient

from common.models import Trade, Ticker
from common.msg_bus import Publisher
from data_service.orderbook import OrderBookManager

logger = logging.getLogger(__name__)


def _parse_ts(ts_str: str) -> float:
    """Parse ISO 8601 timestamp to epoch seconds."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


class DataFeed:
    """Manages WebSocket connection and publishes normalized data."""

    def __init__(
        self,
        symbols: list[str],
        publisher: Publisher,
        api_key: str = "",
        api_secret: str = "",
        key_file: str = "",
        channels: list[str] | None = None,
    ) -> None:
        self._symbols = symbols
        self._pub = publisher
        self._api_key = api_key
        self._api_secret = api_secret
        self._key_file = key_file
        self._channels = channels or ["level2", "market_trades", "ticker"]
        self._books = OrderBookManager()
        self._ws: WSClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._msg_count = 0

    @property
    def orderbooks(self) -> OrderBookManager:
        return self._books

    @property
    def msg_count(self) -> int:
        return self._msg_count

    def _on_message(self, msg) -> None:
        """Callback for incoming WebSocket messages. Runs in WS thread."""
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
                elif channel == "ticker":
                    self._handle_ticker(event)
        except Exception:
            logger.exception("Error processing message")

    def _handle_l2(self, event: dict, raw: dict) -> None:
        """Handle level2 order book updates."""
        event_type = event.get("type", "")
        ts = _parse_ts(raw.get("timestamp", ""))

        for update in event.get("updates", []):
            symbol = update.get("product_id", "")
            if not symbol:
                # Sometimes product_id is at event level
                symbol = event.get("product_id", "")
            book = self._books.get_or_create(symbol)

            side = update.get("side", "").lower()
            if side in ("offer", "ask", "sell"):
                side = "ask"
            else:
                side = "bid"
            price = update.get("price_level", update.get("price", "0"))
            size = update.get("new_quantity", update.get("quantity", "0"))
            book.apply_update(side, price, size)

        # Publish snapshot for each affected symbol
        seen: set[str] = set()
        for update in event.get("updates", []):
            sym = update.get("product_id", event.get("product_id", ""))
            if sym and sym not in seen:
                seen.add(sym)
                ob = self._books.get(sym)
                if ob:
                    snap = ob.snapshot(depth=20, exchange_ts=ts)
                    # Fire-and-forget publish (runs from sync callback)
                    self._loop.call_soon_threadsafe(
                        lambda s=snap: asyncio.ensure_future(
                            self._pub.publish(f"orderbook.{s.symbol}", s.to_dict())
                        )
                    )

    def _handle_trades(self, event: dict) -> None:
        """Handle market trade messages."""
        for t in event.get("trades", []):
            trade = Trade(
                symbol=t.get("product_id", ""),
                price=float(t.get("price", 0)),
                size=float(t.get("size", 0)),
                side=t.get("side", "unknown").lower(),
                trade_id=str(t.get("trade_id", "")),
                exchange_ts=_parse_ts(t.get("time", "")),
            )
            self._loop.call_soon_threadsafe(
                lambda tr=trade: asyncio.ensure_future(
                    self._pub.publish(f"trade.{tr.symbol}", tr.to_dict())
                )
            )

    def _handle_ticker(self, event: dict) -> None:
        """Handle ticker messages."""
        for t in event.get("tickers", []):
            ticker = Ticker(
                symbol=t.get("product_id", ""),
                price=float(t.get("price", 0)),
                volume_24h=float(t.get("volume_24_h", 0)),
                low_24h=float(t.get("low_24_h", 0)),
                high_24h=float(t.get("high_24_h", 0)),
                best_bid=float(t.get("best_bid", 0)),
                best_ask=float(t.get("best_ask", 0)),
                exchange_ts=_parse_ts(t.get("time", "")),
            )
            self._loop.call_soon_threadsafe(
                lambda tk=ticker: asyncio.ensure_future(
                    self._pub.publish(f"ticker.{tk.symbol}", tk.to_dict())
                )
            )

    async def start(self) -> None:
        """Connect to Coinbase WS and subscribe to channels."""
        self._loop = asyncio.get_running_loop()
        logger.info("Starting data feed for %s", self._symbols)

        ws_kwargs: dict = {"on_message": self._on_message}
        if self._key_file:
            ws_kwargs["key_file"] = self._key_file
        elif self._api_key:
            ws_kwargs["api_key"] = self._api_key
            ws_kwargs["api_secret"] = self._api_secret

        self._ws = WSClient(**ws_kwargs)
        self._ws.open()

        # Subscribe to each channel
        for channel in self._channels:
            self._ws.subscribe(
                product_ids=self._symbols,
                channels=[channel],
            )
            logger.info("Subscribed to %s for %s", channel, self._symbols)

        self._running = True
        logger.info("Data feed running.")

    async def run(self) -> None:
        """Keep the feed alive, handling reconnection."""
        await self.start()
        try:
            while self._running:
                try:
                    # Run the blocking WS loop in a thread so asyncio can
                    # still process signals and cancellation.
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, self._ws.run_forever_with_exception_check
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("WebSocket error: %s. Reconnecting...", e)
                    await asyncio.sleep(1)
                    await self.start()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("Data feed stopped. Total messages: %d", self._msg_count)
