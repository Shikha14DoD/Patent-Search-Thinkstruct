"""Search speed on a large index: Part 1's exact search against the engine's HNSW.

    python benchmarks/search_large.py --index index_large

Uses the search engine itself, so it measures the code that ships. Two query
sets: Part 1's 12 benchmark queries (mostly vehicles) and 8 queries from other
fields in the large data. Filter scenarios: Part 1's, plus the two most common
classification classes outside vehicles in the index. For each, three methods:

  exact   Part 1's search (prefilter or mask), every vector scored
  hnsw    the HNSW graph built by build_index, filters applied inside it
  auto    the engine's default: HNSW with no filter, exact search otherwise

Two latencies, medians over queries:

  retrieval  the search alone (query already embedded, evidence not fetched)
  complete   what a user waits for once the model is loaded: embedding the
             query, searching, and fetching up to 3 evidence passages per result

Correctness, measured against exact search, never judged: recall@10 (the same
patents in the top 10) and evidence agreement (the same evidence passages for
each patent both methods return). Writes benchmarks/large/search.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.timing import QUERIES  # noqa: E402
from src.search import Filters, PatentSearch  # noqa: E402

NEW_DOMAIN_QUERIES = [
    "wearable device that monitors heart rate",
    "battery electrode with improved charging capacity",
    "wireless network that allocates bandwidth between users",
    "pharmaceutical composition for treating inflammation",
    "semiconductor transistor with reduced leakage current",
    "machine learning model for classifying images",
    "surgical instrument for minimally invasive procedures",
    "display panel with organic light emitting diodes",
]
BASE_SCENARIOS = [
    ("No filter", Filters()),
    ("Classification B60 (vehicles, broad)", Filters(classification="B60")),
    ("Classification B60C (tyres)", Filters(classification="B60C")),
    ("Title contains 'wheel'", Filters(title_keywords="wheel")),
    ("Classification A61B (medical, selective)", Filters(classification="A61B")),
]
METHODS = ("exact", "hnsw", "auto")


def new_domain_filters(engine: PatentSearch, count: int = 2) -> list[tuple[str, Filters]]:
    """The most common 3-character classification classes outside vehicles (B60)."""
    classes = collections.Counter(
        p["classification"][:3].upper() for p in engine.patents.values()
        if p.get("classification") and not p["classification"].upper().startswith("B60"))
    return [(f"Classification {code} (other field, {n} patents)", Filters(classification=code))
            for code, n in classes.most_common(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact vs HNSW search on a large index.")
    parser.add_argument("--index", default="index_large")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/large/search.json")
    args = parser.parse_args()

    engine = PatentSearch(args.index)
    if not engine.has_hnsw:
        raise SystemExit(f"{args.index} has no HNSW graph: rebuild it with build_index "
                         "(the graph is built automatically for 50,000 vectors or more)")
    engine.hnsw                                   # load the graph once, outside the timings
    engine.model                                  # likewise the model
    scenarios = BASE_SCENARIOS + new_domain_filters(engine)
    print(f"{len(engine.patents)} patents, {len(engine.vectors)} passage vectors, "
          f"HNSW graph and model loaded")

    def retrieval(qv, filters, method):
        report = engine.search(filters=filters, top_k=10, query_vector=qv,
                               strategy=method, max_excerpts=1)
        ms = report.timings_ms["total_ms"] - report.timings_ms.get("evidence_ms", 0.0)
        return [r.patent_id for r in report.results], ms, report.strategy or "no match"

    def complete(query, filters, method):
        start = time.perf_counter()
        report = engine.search(query, filters=filters, top_k=10, strategy=method)
        ms = (time.perf_counter() - start) * 1000
        evidence = {r.patent_id: tuple(e.label for e in r.excerpts) for r in report.results}
        return ms, evidence

    results = {}
    for set_name, queries in (("Part 1 queries (vehicles)", QUERIES),
                              ("New-field queries", NEW_DOMAIN_QUERIES)):
        vectors = [engine.embed_query(q) for q in queries]
        rows = []
        for name, filters in scenarios:
            allowed, _, _ = engine.resolve_filters(filters)
            share = 1.0 if allowed is None else len(allowed) / len(engine.patents)
            ret = {m: [] for m in METHODS}
            recall = {m: [] for m in METHODS}
            full = {m: [] for m in ("exact", "auto")}
            same_evidence, used = [], set()
            for query, qv in zip(queries, vectors):
                truth = None
                for method in METHODS:
                    runs = [retrieval(qv, filters, method) for _ in range(args.repeats)]
                    found = runs[-1][0]
                    ret[method].append(statistics.median(ms for _, ms, _ in runs))
                    truth = found if method == "exact" else truth
                    recall[method].append((1.0 if not found else 0.0) if not truth else
                                          len(set(found) & set(truth)) / len(truth))
                    if method == "auto":
                        used.add(runs[-1][2])
                evidence = {}
                for method in ("exact", "auto"):
                    runs = [complete(query, filters, method) for _ in range(args.repeats)]
                    full[method].append(statistics.median(ms for ms, _ in runs))
                    evidence[method] = runs[-1][1]
                shared = set(evidence["exact"]) & set(evidence["auto"])
                same_evidence.append(1.0 if not shared else
                                     sum(evidence["exact"][p] == evidence["auto"][p]
                                         for p in shared) / len(shared))
            row = {"scenario": name, "patents_allowed_share": round(share, 3),
                   "auto_used": sorted(used),
                   "same_evidence": round(statistics.mean(same_evidence), 3)}
            for method in METHODS:
                row[method] = {"retrieval_ms": round(statistics.median(ret[method]), 2),
                               "recall_at_10": round(statistics.mean(recall[method]), 3),
                               "min_recall_at_10": round(min(recall[method]), 2)}
            for method in ("exact", "auto"):
                row[method]["complete_ms"] = round(statistics.median(full[method]), 2)
            rows.append(row)
            print(f"  {set_name}: {name}: done", flush=True)
        results[set_name] = rows

    print(f"\nMedians over each query set; {args.repeats} repeats. "
          "Auto uses HNSW only without filters.")
    for set_name, rows in results.items():
        print(f"\n### {set_name}\n")
        print("| Scenario | Patents allowed | Retrieval: exact → auto (ms) | Speed-up | "
              "Complete query: exact → auto (ms) | Speed-up | Auto recall@10 | "
              "Same evidence | Auto used | HNSW forced: recall@10 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for r in rows:
            e, a, h = r["exact"], r["auto"], r["hnsw"]
            print(f"| {r['scenario']} | {r['patents_allowed_share']:.0%} | "
                  f"{e['retrieval_ms']} → {a['retrieval_ms']} | "
                  f"{e['retrieval_ms'] / max(a['retrieval_ms'], 1e-9):.1f}× | "
                  f"{e['complete_ms']} → {a['complete_ms']} | "
                  f"{e['complete_ms'] / max(a['complete_ms'], 1e-9):.1f}× | "
                  f"{a['recall_at_10']:.0%} (min {a['min_recall_at_10']:.0%}) | "
                  f"{r['same_evidence']:.0%} | {'/'.join(r['auto_used'])} | "
                  f"{h['recall_at_10']:.0%} (min {h['min_recall_at_10']:.0%}) |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"index": args.index, "patents": len(engine.patents),
                   "vectors": int(len(engine.vectors)), "repeats": args.repeats,
                   "auto_rule": "HNSW without filters, exact otherwise",
                   "query_sets": results}, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
