"""
Step 5 (Phase 2, piece 1): a "fuzzy" cache using text similarity, so tasks
that are near-duplicates -- not just byte-identical -- still hit the
cache. This seems like the obviously better version of caching: why only
catch EXACT repeats when you could catch "almost the same" ones too?

Build this, run the demo at the bottom, and look closely at the output
before reading any further comments about it.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FuzzySimilarityCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self._keys: list[str] = []
        self._answers: list[float] = []
        # TF-IDF over CHARACTER n-grams (not words) -- appropriate for
        # short strings like "3 + 4 * (2 - 1)" where word-level tokens
        # would mostly just be single characters anyway.
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = None
        self.hits = 0
        self.misses = 0

    def _rebuild_index(self):
        self._matrix = self._vectorizer.fit_transform(self._keys) if self._keys else None

    def lookup(self, expression: str) -> float | None:
        if not self._keys:
            self.misses += 1
            return None
        query_vec = self._vectorizer.transform([expression])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        best_idx = sims.argmax()
        if sims[best_idx] >= self.threshold:
            self.hits += 1
            return self._answers[best_idx]
        self.misses += 1
        return None

    def store(self, expression: str, answer: float) -> None:
        self._keys.append(expression)
        self._answers.append(answer)
        self._rebuild_index()


if __name__ == "__main__":
    cache = FuzzySimilarityCache(similarity_threshold=0.85)
    cache.store("3 + 4 * (2 - 1)", 7.0)  # true answer: 3 + 4*1 = 7

    # Now look up a DIFFERENT expression -- one character different, a
    # flipped sign inside the parens:
    result = cache.lookup("3 + 4 * (2 + 1)")  # true answer: 3 + 4*3 = 15
    print(f"cache returned: {result}")
    print(f"but the TRUE answer to '3 + 4 * (2 + 1)' is 15.0")
