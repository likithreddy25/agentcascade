"""
Step 2: the two solver tiers.

NaiveTier: fast, reads left to right, ignores parentheses and operator
precedence entirely. This is our stand-in for "a small/cheap model that's
fine on simple stuff, confidently wrong on structurally complex stuff."

ExactTier: always correct (uses Python's real expression grammar). Our
stand-in for "the expensive model you don't want to call on every task."

Real routing systems route between two LLMs. We're routing between two
deterministic functions instead, on purpose: it lets us get 100% certain,
computed ground truth (did the cheap tier actually get it right?) with zero
API calls, so we can build and debug the ROUTING LOGIC itself before ever
touching a real model.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Made-up but realistic cost ratio: roughly what a small vs. mid-tier LLM
# actually costs per call, at the time this was written. The exact number
# matters less than the SHAPE of the tradeoff it creates later.
NAIVE_COST_UNITS = 1.0
EXACT_COST_UNITS = 12.0


@dataclass
class SolveResult:
    answer: float | None
    success: bool          # did it produce a syntactically valid answer at all
    correct: bool | None    # only known when true_answer is supplied
    cost_units: float


class SolverTier(ABC):
    """
    Why an abstract base class here, instead of just two plain functions?
    Because everything downstream (the router, the orchestrator you'll
    build later) should be able to treat "some solver" generically --
    call .solve(), get an answer back -- without caring whether it's
    NaiveTier, ExactTier, or (eventually) a real Claude model. That's the
    whole point of an interface: it lets you swap the implementation
    later without touching the code that uses it.
    """
    name: str
    cost_units: float

    @abstractmethod
    def solve(self, expression: str) -> tuple[float | None, bool]:
        """Must return (answer, did_it_produce_a_valid_answer)."""
        ...

    def solve_and_score(self, expression: str, true_answer: float | None = None) -> SolveResult:
        answer, ok = self.solve(expression)
        correct = None
        if true_answer is not None and ok:
            correct = abs(answer - true_answer) < 1e-6
        return SolveResult(answer=answer, success=ok, correct=correct,
                            cost_units=self.cost_units)


# Matches either a number (int or decimal) or one of the four operator/
# parenthesis characters. This is a simple TOKENIZER -- it turns a string
# like "3 + (4)" into a flat list of meaningful pieces: ["3", "+", "(",
# "4", ")"], throwing away whitespace, which we don't care about.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[()+\-*/]")


class NaiveTier(SolverTier):
    name = "naive"
    cost_units = NAIVE_COST_UNITS

    def solve(self, expression: str) -> tuple[float | None, bool]:
        try:
            # Strip parentheses entirely rather than respecting them --
            # this IS the naivety. "(3 + 4) * 2" becomes "3 + 4 * 2" to
            # this evaluator, and it will compute that strictly left to
            # right: ((3 + 4) * 2) is NOT what it does -- it does
            # (3 + 4) then * 2 in sequence, same as correct here by luck,
            # but "3 + (4 * 2)" vs "(3 + 4) * 2" is where it breaks, since
            # after stripping parens both look like "3 + 4 * 2" to it.
            stripped = expression.replace("(", "").replace(")", "")
            tokens = _TOKEN_RE.findall(stripped)
            if not tokens:
                return None, False

            total = float(tokens[0])
            i = 1
            # Walk the token list two-at-a-time: operator, then operand,
            # applying immediately to `total`. No precedence, no lookahead
            # -- whatever operator comes next just gets applied right now.
            while i < len(tokens) - 1:
                op, val = tokens[i], float(tokens[i + 1])
                if op == "+":
                    total += val
                elif op == "-":
                    total -= val
                elif op == "*":
                    total *= val
                elif op == "/":
                    if val == 0:
                        return None, False
                    total /= val
                else:
                    return None, False
                i += 2
            return total, True
        except (ValueError, IndexError, ZeroDivisionError):
            return None, False


class ExactTier(SolverTier):
    name = "exact"
    cost_units = EXACT_COST_UNITS

    def solve(self, expression: str) -> tuple[float | None, bool]:
        try:
            # Same safety caveat as tasks.py: eval() is fine here because
            # WE generated this string from a fixed, trusted grammar.
            answer = eval(expression, {"__builtins__": {}}, {})
            return float(answer), True
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
            return None, False


if __name__ == "__main__":
    naive, exact = NaiveTier(), ExactTier()
    for expr in ["3 + 4 * 2", "(3 + 4) * 2", "3 + (4 * 2)", "(10 - 2) / (2 + 2)"]:
        n = naive.solve_and_score(expr)
        e = exact.solve_and_score(expr)
        agree = "SAME" if n.answer == e.answer else "DIFFER"
        print(f"{expr:25s} naive={n.answer!s:10s} exact={e.answer!s:10s} [{agree}]")
