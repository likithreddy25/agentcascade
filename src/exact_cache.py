"""
Step 6 (Phase 2, piece 2): the fix. For an exact-answer domain, the cache
should only return an answer for an expression it has ACTUALLY seen before
-- not "something similar." Normalize whitespace (so "3+4" and "3 + 4"
still count as the same key) and use a plain dict lookup.
"""
from __future__ import annotations

import re


def _normalize(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


class ExactMatchCache:
    def __init__(self):
        self._store: dict[str, float] = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, expression: str) -> float | None:
        key = _normalize(expression)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def store(self, expression: str, answer: float) -> None:
        self._store[_normalize(expression)] = answer

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


if __name__ == "__main__":
    cache = ExactMatchCache()
    cache.store("3 + 4 * (2 - 1)", 7.0)

    # Same expression, different whitespace -- SHOULD hit.
    print("exact match, different spacing:", cache.lookup("3+4*(2-1)"))

    # The near-duplicate that broke the fuzzy version -- MUST miss.
    print("near-duplicate (should be None):", cache.lookup("3 + 4 * (2 + 1)"))
