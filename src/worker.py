"""
Step 11: consumes tasks from Kafka, runs them through the real Orchestrator
(cache -> router -> tier selection), publishes results to another topic.

Run:  python -m src.worker --bootstrap-servers localhost:9092 --max-messages 500
"""
from __future__ import annotations

import argparse

from src.orchestrator import Orchestrator
from src.transport import KafkaConsumer, KafkaProducer

INPUT_TOPIC = "agent_tasks"
OUTPUT_TOPIC = "task_results"


def run(consumer, producer, orchestrator: Orchestrator,
        max_messages: int | None = None, idle_timeout: float = 2.0) -> int:
    # A fresh consumer group has to complete a join-group + partition-
    # assignment handshake with the broker before any message arrives --
    # that alone can take several seconds on a just-started broker. 5
    # quick idle polls (5 seconds total) was found (by actually running
    # this against real Kafka) to be too impatient and could give up
    # before the handshake even finished. 30 polls at 2s each = up to a
    # full minute of patience before concluding the producer is done.
    processed, idle_polls, max_idle_polls = 0, 0, 30
    while max_messages is None or processed < max_messages:
        msg = consumer.poll(timeout=idle_timeout)
        if msg is None:
            idle_polls += 1
            if max_messages is not None and idle_polls >= max_idle_polls:
                # No new messages for a while and we have a cap -- assume
                # the producer is done and stop, rather than waiting forever.
                break
            continue
        idle_polls = 0

        outcome = orchestrator.process(msg["expression"], msg.get("true_answer"))
        producer.send(OUTPUT_TOPIC, key=msg["task_id"], value={
            "task_id": msg["task_id"], "expression": outcome.expression,
            "final_answer": outcome.final_answer, "correct": outcome.correct,
            "source": outcome.source, "cost_units": outcome.cost_units,
        })
        processed += 1
    producer.flush()
    return processed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-servers", default="localhost:9092")
    ap.add_argument("--group-id", default="cascade-worker")
    ap.add_argument("--max-messages", type=int, default=None)
    args = ap.parse_args()

    consumer = KafkaConsumer(args.bootstrap_servers, args.group_id, [INPUT_TOPIC])
    producer = KafkaProducer(args.bootstrap_servers)
    orchestrator = Orchestrator()
    n = run(consumer, producer, orchestrator, max_messages=args.max_messages)
    print(f"Processed {n} tasks. Cache hits: {orchestrator.n_cache_hits}, "
          f"escalations: {orchestrator.n_escalations}, "
          f"total cost: {orchestrator.total_cost_units}")
    consumer.close()
