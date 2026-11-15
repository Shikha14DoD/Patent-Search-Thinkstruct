"""Automated relevance check: can the search find a patent from its own claim?

For every patent, its first claim is used as the query and that patent's
claims are hidden from the search, so it can only be found through its
abstract and description. We record where the patent ranks among all 640,
for the semantic search and for the BM25 keyword baseline, and report:

  Recall@1    share of queries where the patent ranks first
  Recall@10   share where it is in the top 10
  MRR         mean of 1 / rank (higher is better; 1.0 = always first)

A third row, "combined", blends the two rankings with reciprocal rank fusion (RRF): each
patent scores 1/(60 + its semantic rank) + 1/(60 + its BM25 rank). 60 is the
standard constant from the original RRF paper; it is not tuned on this test,
because tuning on the test you report would overstate the result.

What it measures: whether the search connects the legal language of a claim
to the plain description of the same invention. It is a proxy, not a measure
of prior-art relevance. Descriptions often restate claims word for word,
which favours keyword matching, so the test is if anything tilted towards
BM25.

Patents whose abstract is identical to another patent's are skipped: their
twin's claims stay visible and would be found first, which is not a miss.

Usage:
    python evaluation/self_retrieval.py            # writes evaluation/self_retrieval.md
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.bm25 import BM25  # noqa: E402
from src.search import PatentSearch  # noqa: E402
from src.text_utils import tokenize  # noqa: E402

# A query word counts as distinctive when it is rare in the corpus (BM25 idf
# above this value), so common words like "tire" or "comprising" are ignored.
DISTINCTIVE_IDF = 2.0


def first_claim_rows(engine: PatentSearch) -> dict[str, int]:
    """Row of each patent's first claim (lowest claim number, first part)."""
    first: dict[str, int] = {}
    for row, unit in enumerate(engine.units):
        if unit["kind"] != "claim":
            continue
        current = first.get(unit["patent_id"])
        if current is None or unit["position"] < engine.units[current]["position"]:
            first[unit["patent_id"]] = row
    return first


RRF_K = 60


def best_per_patent(engine: PatentSearch, scores: np.ndarray, patent_id: str,
                    zero_is_miss: bool = False) -> np.ndarray:
    """Each patent's best passage score, with patent_id's own claims hidden.

    For BM25 a score of 0 means no word in common, so that patent is not a
    result at all (-inf).
    """
    scores = scores.astype(np.float32, copy=True)
    start, end = engine.unit_rows_of(patent_id)
    for row in range(start, end):
        if engine.units[row]["kind"] == "claim":
            scores[row] = -np.inf
    best = np.maximum.reduceat(scores, engine.block_starts)
    if zero_is_miss:
        best = np.where(best > 0, best, -np.inf)
    return best


def rank_of(best: np.ndarray, position: int) -> int | None:
    """1-based rank of one patent; None if it isn't a result."""
    target = best[position]
    if not np.isfinite(target):
        return None
    return int(np.count_nonzero(best > target)) + 1


def target_rank(engine: PatentSearch, scores: np.ndarray, patent_id: str,
                zero_is_miss: bool = False) -> int | None:
    """Rank of patent_id when its own claims are hidden; None if it can't be found."""
    best = best_per_patent(engine, scores, patent_id, zero_is_miss)
    return rank_of(best, engine.patent_position[patent_id])


def rrf(*bests: np.ndarray) -> np.ndarray:
    """Reciprocal rank fusion of several per-patent score arrays."""
    fused = np.zeros_like(bests[0], dtype=np.float64)
    for best in bests:
        order = np.argsort(-best, kind="stable")
        ranks = np.empty(len(best), dtype=np.float64)
        ranks[order] = np.arange(1, len(best) + 1)
        fused += np.where(np.isfinite(best), 1.0 / (RRF_K + ranks), 0.0)
    return np.where(fused > 0, fused, -np.inf)


def overlap(engine: PatentSearch, bm25: BM25, query: str, patent_id: str) -> float:
    """Share of the query's distinctive words found in the patent's non-claim text."""
    words = {w for w in tokenize(query) if bm25.idf.get(w, 0.0) > DISTINCTIVE_IDF}
    if not words:
        return 1.0
    start, end = engine.unit_rows_of(patent_id)
    rest: set[str] = set()
    for row in range(start, end):
        if engine.units[row]["kind"] != "claim":
            rest.update(tokenize(engine.units[row]["text"]))
    return len(words & rest) / len(words)


def summarise(ranks: list[int | None]) -> dict[str, float]:
    found = [r for r in ranks if r is not None]
    n = len(ranks)
    return {"recall@1": sum(1 for r in found if r == 1) / n,
            "recall@10": sum(1 for r in found if r <= 10) / n,
            "mrr": sum(1 / r for r in found) / n,
            "median_rank": float(np.median([r if r is not None else len(ranks) for r in ranks]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-retrieval relevance check.")
    parser.add_argument("--index", default="index")
    args = parser.parse_args()

    engine = PatentSearch(args.index)
    bm25 = BM25(engine)

    abstracts = Counter(p["abstract"] for p in engine.patents.values())
    first = first_claim_rows(engine)
    targets = [pid for pid in engine.patent_order
               if pid in first and abstracts[engine.patents[pid]["abstract"]] == 1]
    skipped_twins = sum(1 for pid in first if abstracts[engine.patents[pid]["abstract"]] > 1)
    queries = [engine.units[first[pid]]["display_text"] for pid in targets]

    start = time.perf_counter()
    query_vectors = engine.model.encode(queries, batch_size=32, convert_to_numpy=True,
                                        normalize_embeddings=True).astype(np.float32)
    embed_s = time.perf_counter() - start

    semantic, keyword, hybrid, overlaps = [], [], [], []
    for pid, query, vector in zip(targets, queries, query_vectors):
        overlaps.append(overlap(engine, bm25, query, pid))
        position = engine.patent_position[pid]
        sem_best = best_per_patent(engine, engine.vectors @ vector, pid)
        kw_best = best_per_patent(engine, bm25.scores(query), pid, zero_is_miss=True)
        semantic.append(rank_of(sem_best, position))
        keyword.append(rank_of(kw_best, position))
        hybrid.append(rank_of(rrf(sem_best, kw_best), position))

    results = {"Semantic (all-MiniLM-L6-v2)": summarise(semantic),
               "BM25 keyword baseline": summarise(keyword),
               "Combined (RRF of both)": summarise(hybrid)}
    sem_better = sum(1 for s, k in zip(semantic, keyword)
                     if (s or 10**6) < (k or 10**6))
    bm_better = sum(1 for s, k in zip(semantic, keyword)
                    if (k or 10**6) < (s or 10**6))

    # Split by how much of the claim's distinctive wording the description repeats.
    full = [i for i, o in enumerate(overlaps) if o >= 1.0]
    partial = [i for i, o in enumerate(overlaps) if o < 1.0]

    def bucket(indices: list[int]) -> str:
        cells = []
        for ranks in (semantic, keyword, hybrid):
            picked = [ranks[i] for i in indices]
            cells.append(f"{sum(1 for r in picked if r == 1) / len(picked):.1%}")
        return " | ".join(cells)

    lines = ["# Self-retrieval check: finding a patent from its own claim", "",
             f"Each of {len(targets)} patents was searched with its first claim as the query, "
             "with that patent's claims hidden, so it could only be found through its abstract "
             f"and description. {skipped_twins} patents were skipped because another patent has "
             "an identical abstract. Ranks are out of "
             f"{len(engine.patents)} patents; each patent is ranked by its best passage.", "",
             "| Method | Recall@1 | Recall@10 | MRR | Median rank |",
             "|---|---:|---:|---:|---:|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['recall@1']:.1%} | {m['recall@10']:.1%} "
                     f"| {m['mrr']:.3f} | {m['median_rank']:.0f} |")
    lines += ["",
              f"Per query, semantic search ranked the patent higher in {sem_better} cases and "
              f"BM25 in {bm_better}; the rest were ties.", "",
              "## Split by word overlap", "",
              "How much of the claim's distinctive wording (words rare in the corpus) also "
              "appears in the same patent's abstract and description. Recall@1 per group:", "",
              "| Group | Queries | Semantic | BM25 | Combined |",
              "|---|---:|---:|---:|---:|",
              f"| Description repeats every distinctive word | {len(full)} | {bucket(full)} |",
              f"| Some distinctive words missing | {len(partial)} | {bucket(partial)} |", "",
              f"The combined row fuses both rankings with reciprocal rank fusion (k = {RRF_K}, the "
              "standard value, not tuned on this test).", "",
              "This is a proxy: it shows whether the search connects a claim's legal wording to "
              "the plain description of the same invention, not whether results are relevant "
              "prior art. Descriptions often restate claims word for word, which favours BM25.",
              "", f"Query embedding: {len(queries)} claims in {embed_s:.1f} s."]
    (HERE / "self_retrieval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()