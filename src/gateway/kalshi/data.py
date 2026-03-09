"""Kalshi data gateway.

Connects via authenticated WebSocket for real-time orderbook deltas
and fills.  Falls back to REST polling for market discovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

import websockets

from gateway.base import BookUpdate, DataGateway, FillUpdate, OrderUpdate, PositionUpdate, TradeUpdate
from gateway.kalshi.auth import auth_headers
from gateway.kalshi.client import KalshiRestClient
from gateway.kalshi.orderbook import KalshiOrderBook

logger = logging.getLogger(__name__)


class KalshiDataGateway(DataGateway):
    """WebSocket + REST data feed for Kalshi prediction markets."""

    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    def __init__(
        self,
        key_id: str,
        private_key,
        client: KalshiRestClient | None = None,
        market_discovery_interval: float = 300.0,
    ) -> None:
        self._key_id = key_id
        self._private_key = private_key
        self._client = client or KalshiRestClient(key_id, private_key)
        self._discovery_interval = market_discovery_interval

        self._symbols: list[str] = []      # tickers to subscribe to
        self._books: dict[str, KalshiOrderBook] = {}
        self._running = False
        self._ws = None                    # active WS connection

        self._book_callbacks: list[Callable[[BookUpdate], None]] = []
        self._trade_callbacks: list[Callable[[TradeUpdate], None]] = []
        self._fill_callbacks: list[Callable[[FillUpdate], None]] = []
        self._order_callbacks: list[Callable[[OrderUpdate], None]] = []
        self._position_callbacks: list[Callable[[PositionUpdate], None]] = []

    @property
    def exchange_name(self) -> str:
        return "kalshi"

    def on_book_update(self, callback: Callable[[BookUpdate], None]) -> None:
        self._book_callbacks.append(callback)

    def on_trade(self, callback: Callable[[TradeUpdate], None]) -> None:
        self._trade_callbacks.append(callback)

    def on_fill(self, callback: Callable[[FillUpdate], None]) -> None:
        self._fill_callbacks.append(callback)

    def on_order_update(self, callback: Callable[[OrderUpdate], None]) -> None:
        self._order_callbacks.append(callback)

    def on_position_update(self, callback: Callable[[PositionUpdate], None]) -> None:
        self._position_callbacks.append(callback)

    def get_book(self, ticker: str) -> KalshiOrderBook | None:
        return self._books.get(ticker)

    # -- Market discovery (REST) ---------------------------------------------

    async def discover_markets(
        self,
        series_tickers: list[str],
    ) -> list[str]:
        """Discover active market tickers for given series via REST."""
        loop = asyncio.get_running_loop()
        tickers = []
        for series in series_tickers:
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda s=series: self._client.get_markets(
                        series_ticker=s, status="open", limit=100
                    ),
                )
                for m in resp.get("markets", []):
                    tickers.append(m["ticker"])
            except Exception as e:
                logger.error("Market discovery failed for %s: %s", series, e)
        return tickers

    # -- Connection lifecycle ------------------------------------------------

    async def connect(self) -> None:
        self._running = True
        logger.info("Kalshi data gateway ready")

    async def disconnect(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Kalshi data gateway stopped")

    async def subscribe(self, symbols: list[str]) -> None:
        new_tickers = [s for s in symbols if s not in self._symbols]
        self._symbols = list(set(self._symbols + symbols))
        for sym in symbols:
            if sym not in self._books:
                self._books[sym] = KalshiOrderBook()

        # Send WS subscription for new tickers on the live connection
        if new_tickers and self._ws is not None:
            base_id = len(self._symbols) + 100
            for i, ticker in enumerate(new_tickers):
                sub = {
                    "id": base_id + i,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_ticker": ticker,
                    },
                }
                await self._ws.send(json.dumps(sub))
            logger.info("Kalshi WS subscribed to %d new markets", len(new_tickers))

        logger.info("Kalshi tracking %d markets", len(self._symbols))

    async def run(self) -> None:
        """Run WebSocket loop with reconnection."""
        await self.connect()
        while self._running:
            try:
                headers = auth_headers(
                    self._key_id, self._private_key,
                    "GET", "/trade-api/ws/v2",
                )
                async with websockets.connect(
                    self.WS_URL,
                    extra_headers=headers,
                    ping_interval=20,
                ) as ws:
                    self._ws = ws
                    logger.info("Kalshi WS connected")
                    await self._ws_subscribe(ws)
                    async for raw in ws:
                        if not self._running:
                            break
                        self._handle_message(raw)
                    self._ws = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.error("Kalshi WS error: %s, reconnecting...", e)
                    await asyncio.sleep(5)
        await self.disconnect()

    async def _ws_subscribe(self, ws) -> None:
        """Subscribe to orderbook deltas and fills for tracked markets."""
        for i, ticker in enumerate(self._symbols, start=1):
            sub = {
                "id": i,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_ticker": ticker,
                },
            }
            await ws.send(json.dumps(sub))

        # Subscribe to fills (all markets)
        next_id = len(self._symbols) + 1
        await ws.send(json.dumps({
            "id": next_id,
            "cmd": "subscribe",
            "params": {"channels": ["fill"]},
        }))

        # Subscribe to order updates (all markets)
        next_id += 1
        await ws.send(json.dumps({
            "id": next_id,
            "cmd": "subscribe",
            "params": {"channels": ["user_orders"]},
        }))

        # Subscribe to position updates (all markets)
        next_id += 1
        await ws.send(json.dumps({
            "id": next_id,
            "cmd": "subscribe",
            "params": {"channels": ["market_positions"]},
        }))

        logger.info("Kalshi WS subscribed to %d markets + fills + orders + positions",
                     len(self._symbols))

    # -- Message handling ----------------------------------------------------

    def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Kalshi WS: invalid JSON: %s", raw[:200])
            return

        msg_type = data.get("type")

        try:
            if msg_type in ("orderbook_snapshot", "orderbook_delta"):
                self._handle_book(data)
            elif msg_type == "fill":
                self._handle_fill(data)
            elif msg_type == "user_order":
                self._handle_order_update(data)
            elif msg_type == "market_position":
                self._handle_position_update(data)
            elif msg_type == "error":
                logger.error("Kalshi WS error: %s", data)
        except Exception:
            logger.exception("Kalshi WS: failed to handle %s message", msg_type)

    def _handle_book(self, data: dict) -> None:
        msg_type = data.get("type")
        ob_data = data.get("msg", {})
        ticker = ob_data.get("market_ticker", "")

        if ticker not in self._books:
            self._books[ticker] = KalshiOrderBook()
        book = self._books[ticker]

        if msg_type == "orderbook_snapshot":
            book.apply_snapshot(ob_data)
        elif msg_type == "orderbook_delta":
            side = ob_data.get("side")
            if side in ("yes", "no"):
                book.apply_delta(
                    side,
                    float(ob_data.get("price", 0)),
                    int(ob_data.get("delta", 0)),
                )

        # Fire callback
        yes_bid, yes_ask, mid = book.top()
        if yes_bid and yes_ask:
            bu = BookUpdate(
                exchange="kalshi",
                symbol=ticker,
                bid=yes_bid[0],
                ask=yes_ask[0],
                bid_size=yes_bid[1],
                ask_size=yes_ask[1],
                mid=mid,
                ts=time.time(),
            )
            for cb in self._book_callbacks:
                cb(bu)

    def _handle_fill(self, data: dict) -> None:
        fill_data = data.get("msg", {})
        ticker = fill_data.get("market_ticker", "")

        side = fill_data.get("side", "")
        if side == "yes":
            price = float(fill_data.get("yes_price", 0))
        else:
            no_price = fill_data.get("no_price")
            if no_price is not None and float(no_price) > 0:
                price = float(no_price)
            else:
                yes_price = float(fill_data.get("yes_price", 0))
                price = 100 - yes_price if yes_price > 0 else 0

        if price <= 0:
            logger.error("Fill with no valid price: %s", fill_data)
            return

        fu = FillUpdate(
            exchange="kalshi",
            order_id=fill_data.get("order_id", ""),
            symbol=ticker,
            side=side,
            action=fill_data.get("action", ""),
            price=price,
            size=int(fill_data.get("count", 0)),
            fee=float(fill_data.get("total_fees", 0)),
            is_taker=fill_data.get("is_taker", False),
            ts=time.time(),
        )
        for cb in self._fill_callbacks:
            cb(fu)

    def _handle_order_update(self, data: dict) -> None:
        msg = data.get("msg", {})
        ticker = msg.get("ticker", "")
        side = msg.get("side", "")

        # Parse price — Kalshi sends yes_price_dollars as a dollar amount
        yes_price_dollars = float(msg.get("yes_price_dollars", 0) or 0)
        price_cents = round(yes_price_dollars * 100)

        ou = OrderUpdate(
            exchange="kalshi",
            order_id=msg.get("order_id", ""),
            symbol=ticker,
            side=side,
            price=price_cents,
            initial_count=int(float(msg.get("initial_count_fp", 0) or 0)),
            remaining_count=int(float(msg.get("remaining_count_fp", 0) or 0)),
            fill_count=int(float(msg.get("fill_count_fp", 0) or 0)),
            status=msg.get("status", ""),
            ts=time.time(),
        )
        for cb in self._order_callbacks:
            cb(ou)

    def _handle_position_update(self, data: dict) -> None:
        msg = data.get("msg", {})
        ticker = msg.get("market_ticker", "")

        # Log raw fields once for debugging field name conventions
        logger.debug("Position WS raw: %s", {k: v for k, v in msg.items() if k != "user_id"})

        # Kalshi WS field names: try documented names first, fall back to short names
        # position_fp = contract count, realized_pnl_dollars/fees_paid_dollars = dollar amounts
        pos_raw = msg.get("position_fp") or msg.get("position", 0)
        position = int(float(pos_raw or 0))

        # Dollar fields: Kalshi sends fixed-point dollars (e.g. "1.50" = $1.50)
        # Convert to cents by multiplying by 100
        rpnl_raw = msg.get("realized_pnl_dollars") or msg.get("realized_pnl", 0)
        fees_raw = msg.get("fees_paid_dollars") or msg.get("fees_paid", 0)
        realized_pnl = float(rpnl_raw or 0) * 100  # dollars -> cents
        fees_paid = float(fees_raw or 0) * 100      # dollars -> cents

        pu = PositionUpdate(
            exchange="kalshi",
            symbol=ticker,
            position=position,
            realized_pnl=realized_pnl,
            fees_paid=fees_paid,
            ts=time.time(),
        )
        for cb in self._position_callbacks:
            cb(pu)
