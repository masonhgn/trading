"""Recorder service — captures Coinbase + Kalshi data to Parquet.

Two concurrent loops:
1. Coinbase: subscribes to ZMQ bus from data_service (orderbook, trade, ticker)
2. Kalshi: polls REST API for orderbooks, trades, market snapshots

All data is written to partitioned Parquet files for analysis.

Usage:
    # Run alongside data_service for Coinbase data:
    python -m recorder_service.main

    # Kalshi-only (no data_service needed):
    python -m recorder_service.main --kalshi-only

    # Both (default):
    python -m recorder_service.main --coinbase-addr tcp://localhost:5555
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from recorder_service.storage import ParquetWriter
from recorder_service.kalshi_feed import KalshiFeed

logger = logging.getLogger(__name__)


async def _record_coinbase(writer: ParquetWriter, sub_addr: str) -> None:
    """Subscribe to Coinbase data on ZMQ and record it."""
    from common.msg_bus import Subscriber

    sub = Subscriber(connect_addr=sub_addr, topics=["orderbook.", "trade.", "ticker."])
    logger.info("Recording Coinbase data from %s", sub_addr)

    try:
        async for topic, data in sub.listen():
            ts = time.time()

            if topic.startswith("orderbook."):
                symbol = data.get("symbol", "")
                for i, (price, size) in enumerate(data.get("bids", [])):
                    writer.append("orderbook", "coinbase", {
                        "ts": ts, "source": "coinbase", "symbol": symbol,
                        "side": "bid", "level": i, "price": price, "size": size,
                    })
                for i, (price, size) in enumerate(data.get("asks", [])):
                    writer.append("orderbook", "coinbase", {
                        "ts": ts, "source": "coinbase", "symbol": symbol,
                        "side": "ask", "level": i, "price": price, "size": size,
                    })

            elif topic.startswith("trade."):
                writer.append("trade", "coinbase", {
                    "ts": ts, "source": "coinbase",
                    "symbol": data.get("symbol", ""),
                    "price": data.get("price", 0),
                    "size": data.get("size", 0),
                    "side": data.get("side", ""),
                    "trade_id": str(data.get("trade_id", "")),
                })
    finally:
        sub.close()


async def _record_kalshi(
    writer: ParquetWriter,
    feed: KalshiFeed,
    poll_interval: float,
) -> None:
    """Poll Kalshi and record data."""
    logger.info("Recording Kalshi data (poll every %.1fs)", poll_interval)
    await feed.discover_active_markets()
    logger.info("Tracking %d Kalshi markets", len(feed.active_tickers))

    # Rediscover markets periodically (every 5 min)
    last_discover = time.time()

    while True:
        try:
            # Rediscover active markets periodically
            if time.time() - last_discover > 300:
                await feed.discover_active_markets()
                last_discover = time.time()
                logger.info("Refreshed: tracking %d Kalshi markets", len(feed.active_tickers))

            result = await feed.poll_once()
            ts = result["ts"]

            # Record orderbooks
            for ob in result["orderbooks"]:
                ticker = ob["ticker"]
                book = ob.get("orderbook", {})
                for i, level in enumerate(book.get("yes", [])):
                    price = level[0] if isinstance(level, list) else level.get("price", 0)
                    size = level[1] if isinstance(level, list) else level.get("count", 0)
                    writer.append("orderbook", "kalshi", {
                        "ts": ts, "source": "kalshi", "symbol": ticker,
                        "side": "bid", "level": i,
                        "price": float(price), "size": float(size),
                    })
                for i, level in enumerate(book.get("no", [])):
                    price = level[0] if isinstance(level, list) else level.get("price", 0)
                    size = level[1] if isinstance(level, list) else level.get("count", 0)
                    writer.append("orderbook", "kalshi", {
                        "ts": ts, "source": "kalshi", "symbol": ticker,
                        "side": "ask", "level": i,
                        "price": float(price), "size": float(size),
                    })

            # Record trades
            for t in result["trades"]:
                writer.append("trade", "kalshi", {
                    "ts": ts, "source": "kalshi",
                    "symbol": t.get("ticker", ""),
                    "price": float(t.get("yes_price", t.get("price", 0))),
                    "size": float(t.get("count", t.get("size", 0))),
                    "side": t.get("taker_side", ""),
                    "trade_id": str(t.get("trade_id", "")),
                })

            # Record market snapshots
            for m in result["markets"]:
                writer.append("kalshi_market", "kalshi", {
                    "ts": ts,
                    "ticker": m.get("ticker", ""),
                    "series_ticker": m.get("series_ticker", ""),
                    "yes_bid": float(m.get("yes_bid", 0)),
                    "yes_ask": float(m.get("yes_ask", 0)),
                    "last_price": float(m.get("last_price", 0)),
                    "volume": int(m.get("volume", 0)),
                    "open_interest": int(m.get("open_interest", 0)),
                })

            pass  # status printed by _print_status loop

        except Exception:
            logger.exception("Kalshi poll error")

        await asyncio.sleep(poll_interval)


def _fmt_duration(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


async def _print_status(writer: ParquetWriter, feed: KalshiFeed, interval: float = 10.0) -> None:
    """Print a live status summary to the console."""
    while True:
        await asyncio.sleep(interval)

        uptime = _fmt_duration(writer.uptime_seconds)
        counts = writer.row_counts
        total = writer.total_rows
        buffered = writer.buffered_rows
        files = writer.files_written

        # Build per-source breakdown
        cb_ob = counts.get("coinbase/orderbook", 0)
        cb_tr = counts.get("coinbase/trade", 0)
        ka_ob = counts.get("kalshi/orderbook", 0)
        ka_tr = counts.get("kalshi/trade", 0)
        ka_mk = counts.get("kalshi/kalshi_market", 0)

        lines = [
            "",
            f"  ┌─ Recorder Status ─────────────────────────────────┐",
            f"  │  Uptime: {uptime:<12s}  Total rows: {_fmt_count(total):>8s}      │",
            f"  │  Files written: {files:<5d}  Buffered: {_fmt_count(buffered):>8s}       │",
            f"  ├─ Coinbase ─────────────────────────────────────────┤",
            f"  │  Orderbook: {_fmt_count(cb_ob):>8s}    Trades: {_fmt_count(cb_tr):>8s}        │",
            f"  ├─ Kalshi ───────────────────────────────────────────┤",
            f"  │  Orderbook: {_fmt_count(ka_ob):>8s}    Trades: {_fmt_count(ka_tr):>8s}        │",
            f"  │  Markets:   {_fmt_count(ka_mk):>8s}    Active: {len(feed.active_tickers):>5d}           │",
            f"  │  Polls: {feed.poll_count:<8d}                                │",
            f"  └───────────────────────────────────────────────────┘",
        ]
        print("\n".join(lines), flush=True)


async def _periodic_flush(writer: ParquetWriter, interval: float = 60.0) -> None:
    """Periodically flush all buffers to disk."""
    while True:
        await asyncio.sleep(interval)
        writer.flush_all()


async def run(args: argparse.Namespace) -> None:
    writer = ParquetWriter(
        base_dir=args.data_dir,
        flush_interval=args.flush_interval,
        flush_rows=args.flush_rows,
    )
    kalshi_feed = KalshiFeed(
        series_tickers=args.kalshi_series.split(",") if args.kalshi_series else None,
        poll_interval=args.kalshi_poll_interval,
    )

    tasks = []

    if not args.kalshi_only:
        tasks.append(asyncio.create_task(_record_coinbase(writer, args.coinbase_addr)))

    if not args.coinbase_only:
        tasks.append(asyncio.create_task(
            _record_kalshi(writer, kalshi_feed, args.kalshi_poll_interval)
        ))

    tasks.append(asyncio.create_task(_periodic_flush(writer, args.flush_interval)))
    tasks.append(asyncio.create_task(_print_status(writer, kalshi_feed, interval=10.0)))

    logger.info("Recorder running. Data dir: %s", args.data_dir)
    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        logger.info("Flushing remaining data...")
        writer.flush_all()
        await kalshi_feed.close()
        logger.info("Recorder stopped. Total rows recorded: %d", writer.total_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Market data recorder")
    parser.add_argument("--data-dir", default="data",
                        help="Base directory for Parquet output")
    parser.add_argument("--coinbase-addr", default="tcp://localhost:5555",
                        help="ZMQ address for Coinbase data_service")
    parser.add_argument("--kalshi-series",
                        default="KXBTC15M,KXETH15M,KXSOL15M,KXBTCD,KXETHD,KXSOLD",
                        help="Comma-separated Kalshi series tickers")
    parser.add_argument("--kalshi-poll-interval", type=float, default=5.0,
                        help="Seconds between Kalshi REST polls")
    parser.add_argument("--flush-interval", type=float, default=60.0,
                        help="Seconds between Parquet flushes")
    parser.add_argument("--flush-rows", type=int, default=10_000,
                        help="Flush after this many rows per buffer")
    parser.add_argument("--kalshi-only", action="store_true",
                        help="Only record Kalshi data (no Coinbase)")
    parser.add_argument("--coinbase-only", action="store_true",
                        help="Only record Coinbase data (no Kalshi)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Interrupted. Data has been flushed.")


if __name__ == "__main__":
    main()
