"""ZeroMQ-based message bus for inter-service communication.

Publisher side (data_service):
    bus = Publisher("tcp://*:5555")
    await bus.publish("orderbook.BTC-USD", {"bids": [...], "asks": [...]})

Subscriber side (strategy_service, etc.):
    bus = Subscriber("tcp://localhost:5555", topics=["orderbook.BTC-USD", "trade."])
    async for topic, data in bus.listen():
        ...
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import msgpack
import zmq
import zmq.asyncio


class Publisher:
    """Publishes messages on a ZMQ PUB socket with msgpack serialization."""

    def __init__(self, bind_addr: str = "tcp://*:5555") -> None:
        self._addr = bind_addr
        self._ctx = zmq.asyncio.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 100_000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(self._addr)

    async def publish(self, topic: str, data: dict) -> None:
        """Publish data under a topic. Subscribers filter by topic prefix."""
        payload = msgpack.packb(data, use_bin_type=True)
        await self._sock.send_multipart([topic.encode(), payload])

    def close(self) -> None:
        self._sock.close(linger=0)


class Subscriber:
    """Subscribes to topics on a ZMQ SUB socket."""

    def __init__(
        self,
        connect_addr: str = "tcp://localhost:5555",
        topics: list[str] | None = None,
    ) -> None:
        self._addr = connect_addr
        self._ctx = zmq.asyncio.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 100_000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self._addr)

        if topics:
            for t in topics:
                self._sock.subscribe(t.encode())
        else:
            self._sock.subscribe(b"")  # subscribe to everything

    async def listen(self) -> AsyncIterator[tuple[str, dict]]:
        """Yield (topic, data) tuples as they arrive."""
        while True:
            topic_bytes, payload = await self._sock.recv_multipart()
            topic = topic_bytes.decode()
            data = msgpack.unpackb(payload, raw=False)
            yield topic, data

    def close(self) -> None:
        self._sock.close(linger=0)
