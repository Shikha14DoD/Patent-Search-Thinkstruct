"""A BM25 keyword ranking over the same passages, used as an evaluation baseline.

BM25 scores a passage by how many of the query's words it contains, giving
rare words more weight and discounting long passages. It is the standard
keyword baseline in search evaluation.

To compare it fairly with the semantic search, it uses the same passages,
the same tokenizer (lowercase, stemmed, tyre/tire merged), the same filters,
and the same rule for ranking patents: each patent is ranked by its single
best passage. Only the scoring of a passage differs.

It is not used by the CLI: it exists to answer "does semantic search
actually beat keyword search on these queries?" (see evaluation/).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .search import Filters, PatentSearch
from .text_utils import tokenize

# The usual defaults for BM25: k1 controls how quickly repeated words stop
# adding score, b how strongly long passages are discounted.
K1 = 1.5
B = 0.75


class BM25:
    def __init__(self, engine: PatentSearch):
        self.engine = engine
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(len(engine.units), dtype=np.float32)
        for row, unit in enumerate(engine.units):
            # The same text the semantic model embeds, so both methods see
            # identical passages.
            counts = Counter(tokenize(unit["text"]))
            lengths[row] = sum(counts.values())
            for token, count in counts.items():
                postings.setdefault(token, []).append((row, count))
        self.lengths = lengths
        self.average_length = float(lengths.mean()) if len(lengths) else 0.0
        count = len(engine.units)
        self.postings = {
            token: (np.array([r for r, _ in rows], dtype=np.int64),
                    np.array([c for _, c in rows], dtype=np.float32))
            for token, rows in postings.items()
        }
        self.idf = {token: math.log(1 + (count - len(rows) + 0.5) / (len(rows) + 0.5))
                    for token, rows in postings.items()}

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every passage for this query (0 where no word matches)."""
        total = np.zeros(len(self.engine.units), dtype=np.float32)
        norm = K1 * (1 - B + B * self.lengths / max(self.average_length, 1e-9))
        for token in set(tokenize(query)):
            if token not in self.postings:
                continue
            rows, counts = self.postings[token]
            total[rows] += self.idf[token] * counts * (K1 + 1) / (counts + norm[rows])
        return total

    def search(self, query: str, filters: Filters | None = None, top_k: int = 10):
        """Top patents by their best passage: list of (patent_id, score, best row)."""
        engine = self.engine
        scores = self.scores(query)
        allowed, _, _ = engine.resolve_filters(filters or Filters())
        mask = None
        if allowed is not None:
            mask = np.zeros(len(engine.patent_order), dtype=bool)
            mask[[engine.patent_position[p] for p in allowed if p in engine.patent_position]] = True
        best = np.maximum.reduceat(scores, engine.block_starts)
        best = np.where(best > 0, best, -np.inf)   # a patent with no matching word is not a result
        if mask is not None:
            best = np.where(mask, best, -np.inf)
        ranked = []
        for position in engine._top_k(best, top_k):
            start, end = engine.block_starts[position], engine.block_ends[position]
            row = int(start + np.argmax(scores[start:end]))
            ranked.append((engine.patent_order[position], float(best[position]), row))
        return ranked