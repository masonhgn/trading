"""Parquet-based storage for recorded market data.

Buffers rows in memory and flushes to partitioned Parquet files.
Partition layout: data/{source}/{data_type}/{date}/part-{n}.parquet
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────

ORDERBOOK_SCHEMA = pa.schema([
    ("ts", pa.float64()),            # capture timestamp (epoch seconds)
    ("source", pa.string()),         # "coinbase" or "kalshi"
    ("symbol", pa.string()),         # "BTC-USD" or "KXBTC15M-..."
    ("side", pa.string()),           # "bid" or "ask"
    ("level", pa.int32()),           # 0 = best, 1 = second best, etc.
    ("price", pa.float64()),
    ("size", pa.float64()),
])

TRADE_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("source", pa.string()),
    ("symbol", pa.string()),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("side", pa.string()),           # "buy"/"sell" or "yes"/"no"
    ("trade_id", pa.string()),
])

KALSHI_MARKET_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("ticker", pa.string()),
    ("series_ticker", pa.string()),
    ("yes_bid", pa.float64()),
    ("yes_ask", pa.float64()),
    ("last_price", pa.float64()),
    ("volume", pa.int64()),
    ("open_interest", pa.int64()),
])

SCHEMAS = {
    "orderbook": ORDERBOOK_SCHEMA,
    "trade": TRADE_SCHEMA,
    "kalshi_market": KALSHI_MARKET_SCHEMA,
}


class ParquetWriter:
    """Buffers rows and flushes to Parquet files."""

    def __init__(
        self,
        base_dir: str = "data",
        flush_interval: float = 60.0,
        flush_rows: int = 10_000,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._flush_interval = flush_interval
        self._flush_rows = flush_rows
        self._buffers: dict[str, list[dict]] = {}
        self._last_flush: dict[str, float] = {}
        self._file_counts: dict[str, int] = {}
        self._total_rows = 0
        self._row_counts: dict[str, int] = {}  # per source/type
        self._start_time = time.time()
        self._files_written = 0

    def append(self, data_type: str, source: str, row: dict) -> None:
        """Append a row to the buffer. Auto-flushes when thresholds are hit."""
        key = f"{source}/{data_type}"
        if key not in self._buffers:
            self._buffers[key] = []
            self._last_flush[key] = time.time()
            self._file_counts[key] = 0

        self._buffers[key].append(row)
        self._total_rows += 1
        self._row_counts[key] = self._row_counts.get(key, 0) + 1

        # Check flush conditions
        if (len(self._buffers[key]) >= self._flush_rows or
                time.time() - self._last_flush[key] >= self._flush_interval):
            self.flush(data_type, source)

    def flush(self, data_type: str, source: str) -> None:
        """Flush buffered rows to a Parquet file."""
        key = f"{source}/{data_type}"
        rows = self._buffers.get(key, [])
        if not rows:
            return

        schema = SCHEMAS.get(data_type)
        if not schema:
            logger.warning("No schema for data_type=%s", data_type)
            return

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = self._base_dir / source / data_type / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find the next available part number (avoids overwriting on restart)
        if self._file_counts.get(key, 0) == 0:
            existing = sorted(out_dir.glob("part-*.parquet"))
            if existing:
                last = existing[-1].stem  # e.g. "part-0003"
                try:
                    self._file_counts[key] = int(last.split("-")[1])
                except (IndexError, ValueError):
                    pass

        self._file_counts[key] = self._file_counts.get(key, 0) + 1
        filename = f"part-{self._file_counts[key]:04d}.parquet"
        filepath = out_dir / filename

        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, filepath, compression="snappy")

        self._files_written += 1
        self._buffers[key] = []
        self._last_flush[key] = time.time()

    def flush_all(self) -> None:
        """Flush all buffers."""
        for key in list(self._buffers.keys()):
            source, data_type = key.split("/", 1)
            self.flush(data_type, source)

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def buffered_rows(self) -> int:
        return sum(len(b) for b in self._buffers.values())

    @property
    def row_counts(self) -> dict[str, int]:
        return dict(self._row_counts)

    @property
    def files_written(self) -> int:
        return self._files_written

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time
