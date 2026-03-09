"""Example subscriber - shows how other services consume the data feed.

Run this while data_service is running to see live market data:
    python -m common.example_subscriber
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.msg_bus import Subscriber
from common.models import OrderBookSnapshot, Trade, Ticker


async def main() -> None:
    sub = Subscriber(
        connect_addr="tcp://localhost:5555",
        # Subscribe to all topics (or filter: ["orderbook.BTC-USD", "trade."])
        topics=["orderbook.", "trade.", "ticker."],
    )

    print("Listening for market data on tcp://localhost:5555 ...")
    print("Topics: orderbook.*, trade.*, ticker.*\n")

    count = 0
    async for topic, data in sub.listen():
        count += 1

        if topic.startswith("orderbook."):
            snap = OrderBookSnapshot.from_dict(data)
            latency_us = (time.monotonic() - snap.local_ts) * 1e6
            print(
                f"[{count}] {topic}  "
                f"bid={snap.best_bid}  ask={snap.best_ask}  "
                f"spread={snap.spread:.6f}  "
                f"depth={len(snap.bids)}x{len(snap.asks)}  "
                f"latency={latency_us:.0f}μs"
            )

        elif topic.startswith("trade."):
            trade = Trade.from_dict(data)
            print(
                f"[{count}] {topic}  "
                f"{trade.side.upper()} {trade.size} @ {trade.price}  "
                f"id={trade.trade_id}"
            )

        elif topic.startswith("ticker."):
            ticker = Ticker.from_dict(data)
            print(
                f"[{count}] {topic}  "
                f"price={ticker.price}  "
                f"bid={ticker.best_bid}  ask={ticker.best_ask}  "
                f"vol={ticker.volume_24h:.2f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
