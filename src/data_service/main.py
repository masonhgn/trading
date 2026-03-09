"""Data service entry point.

Runs the WebSocket feed and REST poller, publishing all market data
to the ZMQ message bus for consumption by other services.

Usage:
    python -m data_service.main --symbols BTC-USD ETH-USD --key-file cdp_api_key.json
    python -m data_service.main --symbols BTC-USD --api-key KEY --api-secret SECRET
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from common.msg_bus import Publisher
from data_service.feed import DataFeed
from data_service.rest_poller import RESTPoller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_service")


async def run(args: argparse.Namespace) -> None:
    pub = Publisher(bind_addr=args.pub_addr)
    logger.info("ZMQ publisher bound to %s", args.pub_addr)

    feed = DataFeed(
        symbols=args.symbols,
        publisher=pub,
        api_key=args.api_key or os.environ.get("COINBASE_API_KEY_ID", ""),
        api_secret=args.api_secret or os.environ.get("COINBASE_SECRET", ""),
        key_file=args.key_file,
        channels=args.channels,
    )

    poller = RESTPoller(
        symbols=args.symbols,
        publisher=pub,
        api_key=args.api_key or os.environ.get("COINBASE_API_KEY_ID", ""),
        api_secret=args.api_secret or os.environ.get("COINBASE_SECRET", ""),
        key_file=args.key_file,
        candle_interval_sec=args.candle_interval,
    )

    # Run feed and poller concurrently
    tasks = [
        asyncio.create_task(feed.run()),
        asyncio.create_task(poller.run()),
    ]

    stop_event = asyncio.Event()

    def _shutdown() -> None:
        logger.info("Shutting down...")
        stop_event.set()
        for t in tasks:
            t.cancel()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        await feed.stop()
        poller.stop()
        pub.close()
        logger.info("Data service stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CDE Market Data Service")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD"],
        help="Product IDs to subscribe to",
    )
    parser.add_argument(
        "--channels", nargs="+", default=["level2", "market_trades", "ticker"],
        help="WebSocket channels to subscribe to",
    )
    parser.add_argument("--api-key", default="", help="Coinbase API key")
    parser.add_argument("--api-secret", default="", help="Coinbase API secret")
    parser.add_argument("--key-file", default="cdp_api_key.json", help="Path to CDP API key JSON file")
    parser.add_argument(
        "--pub-addr", default="tcp://*:5555",
        help="ZMQ PUB socket bind address",
    )
    parser.add_argument(
        "--candle-interval", type=float, default=60.0,
        help="REST candle poll interval in seconds",
    )

    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down.")


if __name__ == "__main__":
    main()
