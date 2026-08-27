"""
Step 9: message transport abstraction.

The problem this solves: if src/worker.py (which we'll write next) imports
`confluent_kafka.Consumer` directly and calls its methods everywhere, then
testing worker.py requires an actual running Kafka broker. That's slow,
flaky in CI, and annoying while developing. The fix is a classic pattern
called DEPENDENCY INVERSION: worker.py depends on a small interface
(MessageProducer / MessageConsumer) instead of depending on confluent-kafka
directly. Two implementations satisfy that interface -- a real one backed
by Kafka, and an in-memory one backed by a plain Python queue -- and
worker.py can't tell the difference.

You have real Docker, so you'll actually run the Kafka-backed version
end-to-end, not just the in-memory one -- but keeping the in-memory
version around is still worth it for fast local tests later.
"""
from __future__ import annotations

import json
import queue
from abc import ABC, abstractmethod
from typing import Optional


class MessageProducer(ABC):
    @abstractmethod
    def send(self, topic: str, key: str, value: dict) -> None:
        ...

    @abstractmethod
    def flush(self, timeout: float = 10.0) -> None:
        ...


class MessageConsumer(ABC):
    @abstractmethod
    def poll(self, timeout: float = 1.0) -> Optional[dict]:
        """Return the next message's value as a dict, or None if nothing
        arrived within `timeout` seconds."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Real Kafka implementation
# ---------------------------------------------------------------------------

class KafkaProducer(MessageProducer):
    def __init__(self, bootstrap_servers: str):
        from confluent_kafka import Producer  # deferred import: optional dep
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})

    def _delivery_report(self, err, msg):
        if err is not None:
            print(f"[kafka] delivery failed: {err}")

    def send(self, topic: str, key: str, value: dict) -> None:
        self._producer.produce(
            topic, key=key.encode("utf-8"),
            value=json.dumps(value).encode("utf-8"),
            callback=self._delivery_report,
        )
        self._producer.poll(0)  # triggers delivery callbacks without blocking

    def flush(self, timeout: float = 10.0) -> None:
        self._producer.flush(timeout)


class KafkaConsumer(MessageConsumer):
    def __init__(self, bootstrap_servers: str, group_id: str, topics: list[str]):
        from confluent_kafka import Consumer  # deferred import: optional dep
        self._consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })
        self._consumer.subscribe(topics)

    def poll(self, timeout: float = 1.0) -> Optional[dict]:
        msg = self._consumer.poll(timeout)
        if msg is None or msg.error():
            return None
        return json.loads(msg.value().decode("utf-8"))

    def close(self) -> None:
        self._consumer.close()


# ---------------------------------------------------------------------------
# In-memory implementation (for fast tests, no broker needed)
# ---------------------------------------------------------------------------

class InMemoryBroker:
    def __init__(self):
        self._topics: dict[str, "queue.Queue[dict]"] = {}

    def _topic_queue(self, topic: str) -> "queue.Queue[dict]":
        return self._topics.setdefault(topic, queue.Queue())


class InMemoryProducer(MessageProducer):
    def __init__(self, broker: InMemoryBroker):
        self._broker = broker

    def send(self, topic: str, key: str, value: dict) -> None:
        self._broker._topic_queue(topic).put(value)

    def flush(self, timeout: float = 10.0) -> None:
        pass


class InMemoryConsumer(MessageConsumer):
    def __init__(self, broker: InMemoryBroker, topics: list[str]):
        self._broker = broker
        self._topics = topics

    def poll(self, timeout: float = 1.0) -> Optional[dict]:
        for topic in self._topics:
            q = self._broker._topic_queue(topic)
            if not q.empty():
                return q.get()
        return None

    def close(self) -> None:
        pass
