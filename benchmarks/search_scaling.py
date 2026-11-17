"""How exact and HNSW search times change with index size, using existing vectors.

    python benchmarks/search_scaling.py --index index_large

Takes random subsets of the index's patents (all of each chosen patent's
passages), builds an HNSW graph for each subset, and times both searches over
the same subset with Part 1's 12 queries, unfiltered: exact search scores every
vector and keeps each patent's best passage; HNSW fetches 200 passages (doubled
until 10 patents appear), as the engine does. Recall@10 is measured against
exact search on the same subset.

This uses only vectors already embedded, so it shows a trend up to the current
index size (75,018 passages), not performance at millions of passages. It times
the retrieval step itself; filters, query embedding and evidence are left out.
Writes benchmarks/large/search_scaling.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.timing import QUERIES  # noqa: E402
from src.build_index import HNSW_EF_CONSTRUCTION, HNSW_M  # noqa: E402
from src.search import HNSW_CANDIDATES, PatentSearch  # noqa: E402


def median_ms(run, repeats):
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = run()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), result


def main() -> None:
    import faiss

    parser = argparse.ArgumentParser(description="Search time against index size.")
    parser.add_argument("--index", default="index_large")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/large/search_scaling.json")
    args = parser.parse_args()

    engine = PatentSearch(args.index)
    queries = np.stack([engine.embed_query(q) for q in QUERIES]).astype(np.float32)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(engine.patent_order))
    rows_out = []

    for fraction in args.fractions:
        chosen = np.sort(order[:max(10, round(len(order) * fraction))])
        rows = np.concatenate([np.arange(engine.block_starts[p], engine.block_ends[p])
                               for p in chosen])
        vectors = np.ascontiguousarray(engine.vectors[rows])
        owner = engine.unit_patent[rows]                    # patent position of each row
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])

        start = time.perf_counter()
        graph = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
        graph.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        graph.add(vectors)
        build_s = time.perf_counter() - start

        def exact(q):
            best = np.maximum.reduceat(vectors @ q, starts)
            top = np.argsort(-best, kind="stable")[:10]
            return [int(owner[starts[i]]) for i in top]

        def approximate(q):
            k = HNSW_CANDIDATES
            while True:
                params = faiss.SearchParametersHNSW()
                params.efSearch = min(k, len(vectors))
                _, found = graph.search(q[None, :], min(k, len(vectors)), params=params)
                best: dict[int, int] = {}
                for row in found[0]:
                    if row >= 0:
                        best.setdefault(int(owner[row]), len(best))
                if len(best) >= 10 or k >= len(vectors):
                    return list(best)[:10]
                k *= 2

        exact_ms, ann_ms, recall = [], [], []
        for q in queries:
            ms, truth = median_ms(lambda: exact(q), args.repeats)
            exact_ms.append(ms)
            ms, found = median_ms(lambda: approximate(q), args.repeats)
            ann_ms.append(ms)
            recall.append(len(set(found) & set(truth)) / len(truth))
        row = {"patents": int(len(chosen)), "passages": int(len(vectors)),
               "hnsw_build_s": round(build_s, 2),
               "exact_ms": round(statistics.median(exact_ms), 2),
               "hnsw_ms": round(statistics.median(ann_ms), 2),
               "recall_at_10": round(statistics.mean(recall), 3),
               "min_recall_at_10": round(min(recall), 2)}
        rows_out.append(row)
        print(f"  {row['passages']} passages: done", flush=True)

    print(f"\nUnfiltered retrieval only; median over {len(QUERIES)} queries, "
          f"{args.repeats} repeats.\n")
    print("| Patents | Passages | Exact (ms) | HNSW (ms) | Speed-up | HNSW recall@10 (min) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rows_out:
        print(f"| {r['patents']} | {r['passages']:,} | {r['exact_ms']} | {r['hnsw_ms']} | "
              f"{r['exact_ms'] / r['hnsw_ms']:.1f}× | {r['recall_at_10']:.0%} "
              f"({r['min_recall_at_10']:.0%}) |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"index": args.index, "repeats": args.repeats, "sizes": rows_out},
                  handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
