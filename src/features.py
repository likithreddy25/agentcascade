"""
Step 3: features computable from the expression TEXT ALONE, before solving
it. This constraint is not optional -- at real decision time (should we
trust naive or escalate?) we don't know the answer yet, so any feature
that secretly depends on the answer would be "feature leakage": the model
would look great in testing and be useless in production, because in
production that feature is simply unavailable when you need to decide.

Each feature below is a genuine hypothesis about what makes an expression
hard for NaiveTier specifically -- not just "stuff about the expression"
picked at random. Re-read src/tiers.py's NaiveTier before this if the
"why" below doesn't click.
"""
from __future__ import annotations

import re

FEATURE_COLUMNS = [
    "expr_length",
    "n_operators",
    "n_parens_pairs",
    "max_nesting_depth",
    "has_division",
    "n_multiply_or_divide",
    "n_numbers",
]


def _max_nesting_depth(expression: str) -> int:
    """
    How deeply parentheses are nested, e.g. "((1+2)*3)" has depth 2.
    Why this matters for NaiveTier specifically: naive strips ALL
    parentheses before reading, so the deeper/more meaningful the nesting,
    the more likely stripping them silently changed the grouping naive
    computes vs. the real one.
    """
    depth, max_depth = 0, 0
    for ch in expression:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def expression_to_features(expression: str) -> dict:
    n_open = expression.count("(")
    operators = re.findall(r"[+\-*/]", expression)
    numbers = re.findall(r"\d+(?:\.\d+)?", expression)

    return {
        "expr_length": len(expression),
        # Longer expressions have more chances for structure that trips
        # naive up -- a weak but real signal on its own.
        "n_operators": len(operators),
        # More operators = more places precedence/parens COULD matter.
        "n_parens_pairs": n_open,
        # Raw count of parenthesized groups.
        "max_nesting_depth": _max_nesting_depth(expression),
        # The single strongest hypothesis: deep nesting is where naive
        # (which throws parens away) is most likely to diverge from exact.
        "has_division": int("/" in expression),
        # Division interacting with stripped parens is an easy place for
        # naive to silently produce a very different number, not just a
        # "slightly off" one.
        "n_multiply_or_divide": sum(1 for o in operators if o in "*/"),
        # Standard math precedence puts * and / above + and - -- more of
        # them relative to +/- means precedence has more chances to matter,
        # which is exactly the rule naive doesn't implement.
        "n_numbers": len(numbers),
    }


if __name__ == "__main__":
    for expr in ["3 + 4", "3 + 4 * 2", "(3 + 4) * (5 - (2 / 1))"]:
        print(expr, "->", expression_to_features(expr))
