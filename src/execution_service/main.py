"""Execution service entry point.

Subscribes to order request signals on the ZMQ bus, executes them
via the Coinbase API, and publishes order status updates back.

Topics consumed:
    signal.order.*   — OrderRequest from strategy service

Topics published:
    order.<SYMBOL>   — OrderStatus updates (open, filled, cancelled, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from common.msg_bus import Publisher, Subscriber
from common.models import OrderRequest
from execution_service.order_manager import OrderManager
from execution_service.executor import Executor

logger = logging.getLogger(__name__)


async def _order_listener(
    subscriber: Subscriber,
    executor: Executor,
) -> None:
    """Listen for order signals and execute them."""
    logger.info("Listening for order signals on signal.order.*")
    async for topic, data in subscriber.listen():
        try:
            request = OrderRequest.from_dict(data)
            logger.info("Received order signal: %s %s %s @ %s",
                        request.side, request.size, request.symbol, request.limit_price)
            await executor.submit_order(request)
        except Exception:
            logger.exception("Failed to process order signal: %s", topic)


async def _sync_loop(executor: Executor, om: OrderManager, interval: float = 5.0) -> None:
    """Periodically sync open orders with the exchange."""
    while True:
        await asyncio.sleep(interval)
        for order in om.open_orders:
            await executor.sync_order(order.client_order_id)


async def run(args: argparse.Namespace) -> None:
    pub = Publisher(bind_addr=args.pub_addr)
    sub = Subscriber(
        connect_addr=args.sub_addr,
        topics=["signal.order."],
    )
    om = OrderManager()
    executor = Executor(
        publisher=pub,
        order_manager=om,
        api_key=args.api_key,
        api_secret=args.api_secret,
        key_file=args.key_file,
    )

    tasks = [
        asyncio.create_task(_order_listener(sub, executor)),
        asyncio.create_task(_sync_loop(executor, om, interval=args.sync_interval)),
    ]

    logger.info("Execution service running.")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        pub.close()
        sub.close()
        logger.info("Execution service stopped. Orders: %d open, %d total",
                     len(om.open_orders), len(om.all_orders))


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution service")
    parser.add_argument("--api-key", default=os.environ.get("COINBASE_API_KEY_ID", ""))
    parser.add_argument("--api-secret", default=os.environ.get("COINBASE_SECRET", ""))
    parser.add_argument("--key-file", default=os.environ.get("COINBASE_KEY_FILE", "cdp_api_key.json"))
    parser.add_argument("--pub-addr", default="tcp://*:5556",
                        help="ZMQ PUB address for order status updates")
    parser.add_argument("--sub-addr", default="tcp://localhost:5555",
                        help="ZMQ SUB address to receive order signals from")
    parser.add_argument("--sync-interval", type=float, default=5.0,
                        help="Seconds between order state sync with exchange")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down.")


if __name__ == "__main__":
    main()
