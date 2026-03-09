"""Kalshi REST API poller for market data.

Polls orderbooks, trades, and market snapshots for crypto prediction
markets. No authentication needed for these read-only endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Series tickers for crypto 15-min and hourly markets
CRYPTO_SERIES = [
    "KXBTC15M",   # BTC 15-min up/down
    "KXETH15M",   # ETH 15-min up/down
    "KXSOL15M",   # SOL 15-min up/down
    "KXBTCD",     # BTC hourly above/below
    "KXETHD",     # ETH hourly above/below
    "KXSOLD",     # SOL hourly above/below
]


class KalshiFeed:
    """Polls Kalshi REST API for crypto prediction market data."""

    def __init__(
        self,
        series_tickers: list[str] | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        self._series = series_tickers or CRYPTO_SERIES
        self._poll_interval = poll_interval
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._active_tickers: list[str] = []
        self._poll_count = 0
        self._seen_trade_ids: set[str] = set()
        self._max_seen_trades = 10_000  # cap memory usage

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def discover_active_markets(self) -> list[str]:
        """Find currently active (open) market tickers for our series."""
        session = await self._get_session()
        tickers = []

        for series in self._series:
            try:
                url = f"{BASE_URL}/markets"
                params = {
                    "series_ticker": series,
                    "status": "open",
                    "limit": 10,
                }
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning("Kalshi markets list %d for %s", resp.status, series)
                        continue
                    data = await resp.json()
                    markets = data.get("markets", [])
                    for m in markets:
                        ticker = m.get("ticker", "")
                        if ticker:
                            tickers.append(ticker)
                    logger.info("Found %d active markets for %s", len(markets), series)
            except Exception:
                logger.exception("Failed to discover markets for %s", series)

        self._active_tickers = tickers
        return tickers

    async def fetch_orderbook(self, ticker: str) -> dict | None:
        """Fetch orderbook for a market ticker. Returns raw response."""
        session = await self._get_session()
        try:
            url = f"{BASE_URL}/markets/{ticker}/orderbook"
            params = {"depth": 20}
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return {
                    "ticker": ticker,
                    "orderbook": data.get("orderbook", {}),
                    "ts": time.time(),
                }
        except Exception:
            logger.exception("Failed to fetch orderbook for %s", ticker)
            return None

    async def fetch_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        """Fetch recent trades for a market ticker. Deduplicates across polls."""
        session = await self._get_session()
        try:
            url = f"{BASE_URL}/markets/trades"
            params = {"ticker": ticker, "limit": limit}
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                all_trades = data.get("trades", [])
                # Deduplicate: only return trades we haven't seen before
                new_trades = []
                for t in all_trades:
                    tid = str(t.get("trade_id", ""))
                    if tid and tid not in self._seen_trade_ids:
                        self._seen_trade_ids.add(tid)
                        new_trades.append(t)
                # Evict old IDs if set gets too large
                if len(self._seen_trade_ids) > self._max_seen_trades:
                    self._seen_trade_ids = set(list(self._seen_trade_ids)[-5000:])
                return new_trades
        except Exception:
            logger.exception("Failed to fetch trades for %s", ticker)
            return []

    async def fetch_market(self, ticker: str) -> dict | None:
        """Fetch market details (price, volume, open interest)."""
        session = await self._get_session()
        try:
            url = f"{BASE_URL}/markets/{ticker}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get("market", {})
        except Exception:
            logger.exception("Failed to fetch market %s", ticker)
            return None

    async def poll_once(self) -> dict:
        """Poll all active markets once. Returns collected data."""
        if not self._active_tickers:
            await self.discover_active_markets()

        self._poll_count += 1
        result = {
            "orderbooks": [],
            "trades": [],
            "markets": [],
            "ts": time.time(),
        }

        # Fetch orderbooks and market snapshots concurrently
        tasks_ob = [self.fetch_orderbook(t) for t in self._active_tickers]
        tasks_mkt = [self.fetch_market(t) for t in self._active_tickers]

        ob_results = await asyncio.gather(*tasks_ob, return_exceptions=True)
        mkt_results = await asyncio.gather(*tasks_mkt, return_exceptions=True)

        for ob in ob_results:
            if isinstance(ob, dict):
                result["orderbooks"].append(ob)

        for mkt in mkt_results:
            if isinstance(mkt, dict):
                result["markets"].append(mkt)

        # Fetch trades for each market (can be rate-limited, so sequential)
        for ticker in self._active_tickers[:6]:  # limit to avoid rate limits
            trades = await self.fetch_trades(ticker, limit=50)
            result["trades"].extend(trades)

        return result

    @property
    def poll_count(self) -> int:
        return self._poll_count

    @property
    def active_tickers(self) -> list[str]:
        return self._active_tickers

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
