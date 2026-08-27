"""
Step 10: sends generated tasks onto Kafka.

Run:  python -m src.producer --bootstrap-servers localhost:9092 --n 500
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from src.tasks import generate_tasks
from src.transport import KafkaProducer

TOPIC = "agent_tasks"


def run(producer, n: int, seed: int) -> int:
    tasks = generate_tasks(n=n, seed=seed)
    for t in tasks:
        producer.send(TOPIC, key=t.task_id, value=asdict(t))
    producer.flush()
    return len(tasks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-servers", default="localhost:9092")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    producer = KafkaProducer(args.bootstrap_servers)
    n = run(producer, args.n, args.seed)
    print(f"Sent {n} tasks to topic '{TOPIC}'")
