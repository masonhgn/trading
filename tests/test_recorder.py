"""Tests for the recorder service (storage + models)."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pyarrow.parquet as pq

from recorder_service.storage import ParquetWriter, ORDERBOOK_SCHEMA, TRADE_SCHEMA, KALSHI_MARKET_SCHEMA
from common.models import KalshiMarket


def test_parquet_writer_flush():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ParquetWriter(base_dir=tmpdir, flush_rows=5)

        for i in range(5):
            writer.append("trade", "coinbase", {
                "ts": 1000.0 + i,
                "source": "coinbase",
                "symbol": "BTC-USD",
                "price": 50000.0 + i,
                "size": 0.1,
                "side": "buy",
                "trade_id": f"t-{i}",
            })

        # Should have auto-flushed at 5 rows
        assert writer.total_rows == 5
        assert writer.buffered_rows == 0

        # Check file exists
        parquet_files = list(Path(tmpdir).rglob("*.parquet"))
        assert len(parquet_files) == 1

        table = pq.read_table(parquet_files[0])
        assert len(table) == 5
        assert table.column("symbol")[0].as_py() == "BTC-USD"
        assert table.column("price")[0].as_py() == 50000.0


def test_parquet_writer_manual_flush():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ParquetWriter(base_dir=tmpdir, flush_rows=1000)

        writer.append("orderbook", "kalshi", {
            "ts": 1000.0, "source": "kalshi", "symbol": "KXBTC15M-test",
            "side": "bid", "level": 0, "price": 55.0, "size": 10.0,
        })
        writer.append("orderbook", "kalshi", {
            "ts": 1000.0, "source": "kalshi", "symbol": "KXBTC15M-test",
            "side": "ask", "level": 0, "price": 60.0, "size": 8.0,
        })

        assert writer.buffered_rows == 2
        writer.flush_all()
        assert writer.buffered_rows == 0

        parquet_files = list(Path(tmpdir).rglob("*.parquet"))
        assert len(parquet_files) == 1

        table = pq.read_table(parquet_files[0])
        assert len(table) == 2


def test_parquet_writer_multiple_sources():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ParquetWriter(base_dir=tmpdir, flush_rows=1000)

        writer.append("trade", "coinbase", {
            "ts": 1000.0, "source": "coinbase", "symbol": "BTC-USD",
            "price": 50000.0, "size": 0.1, "side": "buy", "trade_id": "cb-1",
        })
        writer.append("trade", "kalshi", {
            "ts": 1000.0, "source": "kalshi", "symbol": "KXBTC15M-test",
            "price": 60.0, "size": 5.0, "side": "yes", "trade_id": "k-1",
        })

        writer.flush_all()

        parquet_files = list(Path(tmpdir).rglob("*.parquet"))
        assert len(parquet_files) == 2  # separate files for each source


def test_parquet_writer_kalshi_market():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ParquetWriter(base_dir=tmpdir, flush_rows=1000)

        writer.append("kalshi_market", "kalshi", {
            "ts": 1000.0,
            "ticker": "KXBTC15M-26MAR08T1200",
            "series_ticker": "KXBTC15M",
            "yes_bid": 55.0,
            "yes_ask": 58.0,
            "last_price": 56.0,
            "volume": 1200,
            "open_interest": 500,
        })

        writer.flush_all()

        parquet_files = list(Path(tmpdir).rglob("*.parquet"))
        assert len(parquet_files) == 1

        table = pq.read_table(parquet_files[0])
        assert table.column("ticker")[0].as_py() == "KXBTC15M-26MAR08T1200"
        assert table.column("volume")[0].as_py() == 1200


def test_kalshi_market_model():
    mkt = KalshiMarket(
        ticker="KXBTC15M-26MAR08T1200",
        event_ticker="KXBTC15M-26MAR08T1200",
        series_ticker="KXBTC15M",
        title="BTC Up or Down",
        subtitle="15 minutes",
        status="open",
        yes_bid=55.0,
        yes_ask=60.0,
        last_price=57.0,
        volume=1000,
        open_interest=400,
        expiration_ts=1741500000.0,
    )
    assert mkt.mid == 57.5
    assert mkt.spread == 5.0

    d = mkt.to_dict()
    mkt2 = KalshiMarket.from_dict(d)
    assert mkt2.ticker == mkt.ticker
    assert mkt2.yes_bid == 55.0


def test_kalshi_market_empty_book():
    mkt = KalshiMarket(
        ticker="KXBTC15M-test", event_ticker="", series_ticker="KXBTC15M",
        title="", subtitle="", status="open",
        yes_bid=0, yes_ask=0, last_price=0,
        volume=0, open_interest=0, expiration_ts=0,
    )
    assert mkt.mid is None
    assert mkt.spread is None


def test_partition_layout():
    """Verify that files are written in the expected directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ParquetWriter(base_dir=tmpdir, flush_rows=1000)

        writer.append("trade", "coinbase", {
            "ts": 1000.0, "source": "coinbase", "symbol": "ETH-USD",
            "price": 3000.0, "size": 1.0, "side": "sell", "trade_id": "t-1",
        })
        writer.flush_all()

        # Check directory structure: base/coinbase/trade/YYYY-MM-DD/part-0001.parquet
        coinbase_dir = Path(tmpdir) / "coinbase" / "trade"
        assert coinbase_dir.exists()
        date_dirs = list(coinbase_dir.iterdir())
        assert len(date_dirs) == 1
        parquet_files = list(date_dirs[0].glob("*.parquet"))
        assert len(parquet_files) == 1
        assert parquet_files[0].name == "part-0001.parquet"
