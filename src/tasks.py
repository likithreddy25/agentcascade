"""
Step 1: generate random arithmetic expressions with varying structure.

The core idea: we need expressions where SOME are trivially easy (flat,
no parens: "3 + 4") and some are structurally hard for a naive left-to-
right evaluator (nested, parenthesized: "(3 + 4) * (5 - 2)"). That
difficulty gradient is what our router will later learn to predict.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

_OPS = ["+", "-", "*", "/"]


@dataclass
class Task:
    task_id: str
    expression: str
    true_answer: float


def _random_number(rng: random.Random) -> str:
    # Keep numbers small and positive -- makes debugging by eye much easier
    # later ("did 3+4*2 really evaluate to 11?") than if numbers were huge
    # or negative right out of the gate.
    return str(rng.randint(1, 20))


def _random_expr(rng: random.Random, depth: int, max_depth: int) -> str:
    """
    Recursively builds an expression string.

    Why recursive? An expression is either:
      - just a number, OR
      - (some expression) OPERATOR (some expression)

    That's a textbook recursive structure -- "an expression is made of
    smaller expressions" -- so a recursive function is the natural fit.
    `depth` stops it from recursing forever; `rng.random() < 0.35` gives it
    a chance to bottom out EARLY even before max_depth, so we don't only
    generate maximally-deep expressions.
    """
    if depth >= max_depth or rng.random() < 0.35:
        return _random_number(rng)

    left = _random_expr(rng, depth + 1, max_depth)
    right = _random_expr(rng, depth + 1, max_depth)
    op = rng.choice(_OPS)

    # Randomly wrap sub-expressions in parentheses. THIS is what actually
    # creates the "hard for a naive evaluator" cases -- without
    # parentheses, operator precedence still matters (2 + 3 * 4 != 20) but
    # parens let us force grouings a left-to-right reader can't get right
    # any other way.
    if rng.random() < 0.5:
        left = f"({left})"
    if rng.random() < 0.5:
        right = f"({right})"
    return f"{left} {op} {right}"


def generate_tasks(n: int, seed: int, max_depth: int = 4) -> list[Task]:
    """
    Generates n valid tasks. `seed` makes this reproducible -- same seed,
    same tasks, every time. That matters a lot for debugging: if your
    router's accuracy looks wrong, you want to be able to regenerate the
    EXACT same dataset, not a new random one, while you investigate.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    attempts = 0

    # Why a while loop with an attempts cap instead of a simple for loop?
    # Some generated expressions will fail to evaluate (division by zero,
    # or a stray syntax issue) -- we skip those rather than crash, but we
    # cap total attempts so a bad run can't loop forever.
    while len(tasks) < n and attempts < n * 20:
        attempts += 1
        expr = _random_expr(rng, depth=0, max_depth=rng.randint(1, max_depth))
        try:
            # eval() with empty builtins/globals is a safe-enough way to
            # compute ground truth here because we generated the string
            # ourselves from a fixed grammar (numbers + operators + parens
            # only) -- it never sees untrusted input. Don't reuse this
            # pattern on a string that came from a user or the network.
            true_answer = eval(expr, {"__builtins__": {}}, {})
            if not isinstance(true_answer, (int, float)) or abs(true_answer) > 1e9:
                continue
        except ZeroDivisionError:
            continue

        tasks.append(Task(
            task_id=f"task_{len(tasks):06d}",
            expression=expr,
            true_answer=float(true_answer),
        ))
    return tasks


if __name__ == "__main__":
    # Quick manual sanity check -- run this file directly and eyeball the
    # output before trusting it in anything downstream.
    for t in generate_tasks(n=10, seed=1):
        print(f"{t.task_id}: {t.expression} = {t.true_answer}")
