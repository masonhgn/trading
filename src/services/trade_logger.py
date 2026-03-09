"""Structured trade logging to Parquet files.

Records every signal, order, fill, and periodic P&L snapshots
so we can analyze performance after the fact and iterate.

Layout:
    logs/
    ├── signals/YYYY-MM-DD/part-NNNN.parquet
    ├── orders/YYYY-MM-DD/part-NNNN.parquet
    ├── fills/YYYY-MM-DD/part-NNNN.parquet
    └── snapshots/YYYY-MM-DD/part-NNNN.parquet
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────

SIGNAL_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("ticker", pa.string()),
    ("asset", pa.string()),
    ("side", pa.string()),           # "buy_yes" or "buy_no"
    ("price_cents", pa.float64()),   # price we'd pay
    ("fair_value", pa.float64()),    # model fair value in cents
    ("edge", pa.float64()),          # signed edge in cents
    ("spot", pa.float64()),          # current spot price
    ("spot_open", pa.float64()),     # spot at contract window open
    ("spot_return_pct", pa.float64()),
    ("vol_15m", pa.float64()),
    ("time_remaining", pa.float64()),
    ("kalshi_bid", pa.float64()),
    ("kalshi_ask", pa.float64()),
    ("position", pa.int32()),        # current position in this contract
    ("paper", pa.bool_()),
])

ORDER_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("ticker", pa.string()),
    ("asset", pa.string()),
    ("side", pa.string()),           # "bid" or "ask"
    ("price_cents", pa.float64()),
    ("size", pa.int32()),
    ("order_id", pa.string()),
    ("success", pa.bool_()),
    ("error", pa.string()),
    ("paper", pa.bool_()),
])

FILL_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("ticker", pa.string()),
    ("asset", pa.string()),
    ("side", pa.string()),           # "yes" or "no"
    ("action", pa.string()),         # "buy" or "sell"
    ("price_cents", pa.float64()),
    ("size", pa.float64()),
    ("fee_cents", pa.float64()),
    ("order_id", pa.string()),
    ("matched_side", pa.string()),   # "bid" or "ask" or "" if unmatched
])

SNAPSHOT_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("ticker", pa.string()),
    ("asset", pa.string()),
    ("position", pa.int32()),
    ("cash_cents", pa.float64()),    # realized P&L
    ("mtm_cents", pa.float64()),     # mark-to-market unrealized
    ("total_pnl_cents", pa.float64()),
    ("fill_count", pa.int32()),
    ("fees_paid", pa.float64()),
    ("kalshi_mid", pa.float64()),
    ("spot", pa.float64()),
    ("signals_total", pa.int32()),   # global counter
    ("orders_total", pa.int32()),
    ("orders_rejected", pa.int32()),
])

SETTLEMENT_SCHEMA = pa.schema([
    ("ts", pa.float64()),             # when we recorded the settlement
    ("ticker", pa.string()),
    ("asset", pa.string()),
    ("window_end", pa.float64()),     # contract expiration time
    ("entry_side", pa.string()),      # "yes" or "no"
    ("entry_price", pa.float64()),    # avg entry in cents
    ("entry_size", pa.int32()),       # contracts held at settlement
    ("settlement_result", pa.string()),  # "yes" or "no" (which side won)
    ("payout_cents", pa.float64()),   # total payout received
    ("pnl_cents", pa.float64()),      # net P&L after cost + fees
    ("fees_cents", pa.float64()),
    ("fair_value_at_entry", pa.float64()),  # what our model predicted
    ("model_correct", pa.bool_()),    # did our model agree with settlement?
])

SCHEMAS = {
    "signals": SIGNAL_SCHEMA,
    "orders": ORDER_SCHEMA,
    "fills": FILL_SCHEMA,
    "snapshots": SNAPSHOT_SCHEMA,
    "settlements": SETTLEMENT_SCHEMA,
}


class TradeLogger:
    """Append-only structured trade logger writing to Parquet."""

    def __init__(self, base_dir: str = "logs", flush_rows: int = 50) -> None:
        self._base_dir = Path(base_dir)
        self._flush_rows = flush_rows
        self._buffers: dict[str, list[dict]] = {}
        self._file_counts: dict[str, int] = {}

    # -- Public API -------------------------------------------------------

    def log_signal(
        self,
        ts: float,
        ticker: str,
        asset: str,
        side: str,
        price_cents: float,
        fair_value: float,
        edge: float,
        spot: float,
        spot_open: float,
        spot_return_pct: float,
        vol_15m: float,
        time_remaining: float,
        kalshi_bid: float,
        kalshi_ask: float,
        position: int,
        paper: bool,
    ) -> None:
        self._append("signals", {
            "ts": ts,
            "ticker": ticker,
            "asset": asset,
            "side": side,
            "price_cents": price_cents,
            "fair_value": fair_value,
            "edge": edge,
            "spot": spot,
            "spot_open": spot_open,
            "spot_return_pct": spot_return_pct,
            "vol_15m": vol_15m,
            "time_remaining": time_remaining,
            "kalshi_bid": kalshi_bid,
            "kalshi_ask": kalshi_ask,
            "position": position,
            "paper": paper,
        })

    def log_order(
        self,
        ts: float,
        ticker: str,
        asset: str,
        side: str,
        price_cents: float,
        size: int,
        order_id: str,
        success: bool,
        error: str = "",
        paper: bool = False,
    ) -> None:
        self._append("orders", {
            "ts": ts,
            "ticker": ticker,
            "asset": asset,
            "side": side,
            "price_cents": price_cents,
            "size": size,
            "order_id": order_id,
            "success": success,
            "error": error,
            "paper": paper,
        })

    def log_fill(
        self,
        ts: float,
        ticker: str,
        asset: str,
        side: str,
        action: str,
        price_cents: float,
        size: float,
        fee_cents: float,
        order_id: str,
        matched_side: str = "",
    ) -> None:
        self._append("fills", {
            "ts": ts,
            "ticker": ticker,
            "asset": asset,
            "side": side,
            "action": action,
            "price_cents": price_cents,
            "size": size,
            "fee_cents": fee_cents,
            "order_id": order_id,
            "matched_side": matched_side,
        })

    def log_snapshot(
        self,
        ts: float,
        ticker: str,
        asset: str,
        position: int,
        cash_cents: float,
        mtm_cents: float,
        total_pnl_cents: float,
        fill_count: int,
        fees_paid: float,
        kalshi_mid: float,
        spot: float,
        signals_total: int,
        orders_total: int,
        orders_rejected: int,
    ) -> None:
        self._append("snapshots", {
            "ts": ts,
            "ticker": ticker,
            "asset": asset,
            "position": position,
            "cash_cents": cash_cents,
            "mtm_cents": mtm_cents,
            "total_pnl_cents": total_pnl_cents,
            "fill_count": fill_count,
            "fees_paid": fees_paid,
            "kalshi_mid": kalshi_mid,
            "spot": spot,
            "signals_total": signals_total,
            "orders_total": orders_total,
            "orders_rejected": orders_rejected,
        })

    def log_settlement(
        self,
        ts: float,
        ticker: str,
        asset: str,
        window_end: float,
        entry_side: str,
        entry_price: float,
        entry_size: int,
        settlement_result: str,
        payout_cents: float,
        pnl_cents: float,
        fees_cents: float,
        fair_value_at_entry: float,
        model_correct: bool,
    ) -> None:
        self._append("settlements", {
            "ts": ts,
            "ticker": ticker,
            "asset": asset,
            "window_end": window_end,
            "entry_side": entry_side,
            "entry_price": entry_price,
            "entry_size": entry_size,
            "settlement_result": settlement_result,
            "payout_cents": payout_cents,
            "pnl_cents": pnl_cents,
            "fees_cents": fees_cents,
            "fair_value_at_entry": fair_value_at_entry,
            "model_correct": model_correct,
        })

    def flush_all(self) -> None:
        """Flush all buffers to disk."""
        for key in list(self._buffers.keys()):
            self._flush(key)

    # -- Internal ---------------------------------------------------------

    def _append(self, table_name: str, row: dict) -> None:
        if table_name not in self._buffers:
            self._buffers[table_name] = []
        self._buffers[table_name].append(row)

        if len(self._buffers[table_name]) >= self._flush_rows:
            self._flush(table_name)

    def _flush(self, table_name: str) -> None:
        rows = self._buffers.get(table_name, [])
        if not rows:
            return

        schema = SCHEMAS.get(table_name)
        if not schema:
            return

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = self._base_dir / table_name / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find next part number
        if self._file_counts.get(table_name, 0) == 0:
            existing = sorted(out_dir.glob("part-*.parquet"))
            if existing:
                last = existing[-1].stem
                try:
                    self._file_counts[table_name] = int(last.split("-")[1])
                except (IndexError, ValueError):
                    pass

        self._file_counts[table_name] = self._file_counts.get(table_name, 0) + 1
        filename = f"part-{self._file_counts[table_name]:04d}.parquet"
        filepath = out_dir / filename

        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, filepath, compression="snappy")
        self._buffers[table_name] = []
        logger.debug("Flushed %d rows to %s", len(rows), filepath)
