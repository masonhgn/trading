"""Kalshi authenticated REST client.

Thin wrapper around requests that handles auth header generation,
rate limiting, and query string building.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import requests

from gateway.kalshi.auth import auth_headers


class HttpError(Exception):
    def __init__(self, reason: str, status: int):
        super().__init__(reason)
        self.reason = reason
        self.status = status

    def __str__(self) -> str:
        return f"HttpError({self.status} {self.reason})"


class KalshiRestClient:
    """Authenticated REST client for the Kalshi Trade API v2."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, key_id: str, private_key) -> None:
        self.key_id = key_id
        self.private_key = private_key
        self._last_call = datetime.now()
        self._rate_limit_ms = 100

    # -- HTTP primitives -----------------------------------------------------

    def _rate_limit(self) -> None:
        now = datetime.now()
        delta = timedelta(milliseconds=self._rate_limit_ms)
        if now - self._last_call < delta:
            time.sleep(self._rate_limit_ms / 1000)
        self._last_call = datetime.now()

    def _headers(self, method: str, path: str) -> dict:
        return auth_headers(self.key_id, self.private_key, method, "/trade-api/v2" + path)

    def get(self, path: str, params: dict | None = None) -> Any:
        self._rate_limit()
        url = self.BASE_URL + path
        resp = requests.get(url, headers=self._headers("GET", path), params=params or {}, timeout=10)
        self._raise_if_bad(resp)
        return resp.json()

    def post(self, path: str, body: dict) -> Any:
        self._rate_limit()
        url = self.BASE_URL + path
        headers = self._headers("POST", path)
        headers["Content-Type"] = "application/json"
        resp = requests.post(url, json=body, headers=headers, timeout=10)
        self._raise_if_bad(resp)
        return resp.json()

    def delete(self, path: str, params: dict | None = None) -> Any:
        self._rate_limit()
        url = self.BASE_URL + path
        resp = requests.delete(url, headers=self._headers("DELETE", path), params=params or {}, timeout=10)
        self._raise_if_bad(resp)
        return resp.json()

    def _raise_if_bad(self, resp: requests.Response) -> None:
        if resp.status_code not in range(200, 300):
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HttpError(f"{resp.reason} - {detail}", resp.status_code)

    # -- Market data endpoints -----------------------------------------------

    def get_markets(
        self,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict:
        params = {k: v for k, v in {
            "series_ticker": series_ticker,
            "status": status,
            "limit": limit,
            "cursor": cursor,
        }.items() if v is not None}
        return self.get("/markets", params)

    def get_market(self, ticker: str) -> dict:
        return self.get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 20) -> dict:
        return self.get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_trades(
        self,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        params: dict = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return self.get("/markets/trades", params)

    def get_event(self, event_ticker: str) -> dict:
        return self.get(f"/events/{event_ticker}")

    def get_series(self, series_ticker: str) -> dict:
        return self.get(f"/series/{series_ticker}")

    # -- Portfolio / order endpoints -----------------------------------------

    def get_balance(self) -> dict:
        return self.get("/portfolio/balance")

    def get_positions(
        self,
        ticker: str | None = None,
        settlement_status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict:
        params = {k: v for k, v in {
            "ticker": ticker,
            "settlement_status": settlement_status,
            "limit": limit,
            "cursor": cursor,
        }.items() if v is not None}
        return self.get("/portfolio/positions", params)

    def create_order(
        self,
        ticker: str,
        client_order_id: str,
        side: str,
        action: str,
        count: int,
        order_type: str = "limit",
        yes_price: int | None = None,
        no_price: int | None = None,
    ) -> dict:
        body: dict = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        return self.post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self.delete(f"/portfolio/orders/{order_id}")

    def get_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> dict:
        params = {k: v for k, v in {
            "ticker": ticker,
            "status": status,
            "limit": limit,
        }.items() if v is not None}
        return self.get("/portfolio/orders", params)

    def get_order(self, order_id: str) -> dict:
        return self.get(f"/portfolio/orders/{order_id}")

    def get_fills(
        self,
        ticker: str | None = None,
        order_id: str | None = None,
        limit: int | None = None,
    ) -> dict:
        params = {k: v for k, v in {
            "ticker": ticker,
            "order_id": order_id,
            "limit": limit,
        }.items() if v is not None}
        return self.get("/portfolio/fills", params)
