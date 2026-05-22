"""Message bus abstraction with in-memory and Kafka backends."""

from __future__ import annotations

import asyncio
import json
import os


class InMemoryBus:
    """Asyncio pub/sub for a single process — no broker required.

    Each subscriber gets its own queue; publishing fans a message out to all.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def publish(self, message: dict) -> None:
        for q in list(self._subscribers):
            await q.put(message)

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)


class KafkaBus:
    """Kafka-backed bus (requires a running broker). Lazy imports kafka-python."""

    def __init__(self, bootstrap_servers: str | None = None, topic: str = "quantmind.regime") -> None:
        self.bootstrap = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = topic
        self._producer = None

    def _get_producer(self):
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        return self._producer

    def publish(self, message: dict) -> None:
        self._get_producer().send(self.topic, message)

    def consume(self, timeout_ms: int = 1000):
        """Yield messages from the topic (blocking; run in a worker thread)."""
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            consumer_timeout_ms=timeout_ms,
        )
        for message in consumer:
            yield message.value
