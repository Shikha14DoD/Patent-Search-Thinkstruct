"""Time the same queries with and without filters.

For every (query, filter scenario) pair this runs each filtering strategy
many times and reports the median and 95th-percentile latency, the number
of patents and units that survive the filters, and whether all strategies
returned the same results.

Query embedding is measured separately and left out of the per-strategy
numbers: it is identical for every strategy and several times larger than
the search itself, so including it would hide the differences being
measured.

Usage:
    python benchmarks/timing.py                 # writes benchmarks/results.md and .json
    python benchmarks/timing.py --repeats 50
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search import Filters, PatentSearch  # noqa: E402

# Fixed queries: short keyword-like, natural-language, and one claim-like.
QUERIES = [
    "wheel rim",
    "sensor that detects a flat tire while driving",
    "tread pattern that improves grip on wet roads",
    "rubber composition containing silica for low rolling resistance",
    "lightweight wheel made of carbon fiber composite",
    "non-pneumatic tire with flexible spokes",
    "trailer hitch coupling with automatic locking",
    "wheel hub bearing assembly with integrated seal",
    "reducing tire cavity noise with sound absorbing material",
    "caster wheel with a brake for furniture",
    "tire bead reinforcement structure",
    "A pneumatic tire comprising a tread portion, a pair of sidewall portions and a pair "
    "of bead portions, wherein a circumferential groove is formed in the tread portion",
]

# Filter scenarios, from broad to very selective. The codes and words are
# specific to the supplied sample; the engine itself hardcodes none of them.
SCENARIOS = [
    ("no filters", Filters()),
    ("classification B60B", Filters(classification="B60B")),
    ("classification B60C11", Filters(classification="B60C11")),
    ("title/abstract keyword: tread", Filters(title_or_abstract_keywords="tread")),
    ("abstract keywords: tread groove", Filters(abstract_keywords="tread groove")),
    ("title/abstract keyword: sensor", Filters(title_or_abstract_keywords="sensor")),
    ("B60C + title keyword: pneumatic",
     Filters(classification="B60C", title_keywords="pneumatic")),
    ('exact title: "PNEUMATIC TIRE"', Filters(exact_title="PNEUMATIC TIRE")),
]
STRATEGIES = ["prefilter", "mask", "postfilter"]
STAGES = ["filter_ms", "score_ms", "rank_ms", "evidence_ms", "total_ms"]


def summarise(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"median": statistics.median(ordered), "p95": p95, "n": len(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark filtered vs unfiltered search.")
    parser.add_argument("--index", default="index")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    start = time.perf_counter()
    engine = PatentSearch(args.index)
    index_load_s = time.perf_counter() - start

    start = time.perf_counter()
    engine.model
    model_load_s = time.perf_counter() - start

    # --- query embedding, measured on its own ----------------------------
    for query in QUERIES[:2]:
        engine.embed_query(query)  # warm-up
    embed_times: list[float] = []
    query_vectors = {}
    for query in QUERIES:
        for _ in range(10):
            start = time.perf_counter()
            vector = engine.embed_query(query)
            embed_times.append((time.perf_counter() - start) * 1000)
        query_vectors[query] = vector
    embed = summarise(embed_times)

    # --- search, per scenario and strategy -------------------------------
    samples: dict[tuple[str, str], dict[str, list[float]]] = {}
    counts: dict[str, dict] = {}
    mismatches: list[str] = []

    for label, filters in SCENARIOS:
        strategies = STRATEGIES if filters.active() else ["mask"]
        for strategy in strategies:
            samples[(label, strategy)] = {stage: [] for stage in STAGES}

        # Candidate counts and a correctness check before timing anything.
        for query in QUERIES:
            outcomes = {}
            for strategy in strategies:
                report = engine.search(filters=filters, top_k=args.top, strategy=strategy,
                                       query_vector=query_vectors[query])
                outcomes[strategy] = [r.patent_id for r in report.results]
                if strategy == strategies[0]:
                    counts[label] = {
                        "eligible_patents": report.candidate_patents,
                        "eligible_units": report.candidate_units,
                        "share_of_units": report.candidate_units / report.total_units,
                        "filter_counts": report.filter_counts,
                    }
            if any(outcomes[s] != outcomes[strategies[0]] for s in strategies):
                mismatches.append(f"{label} / {query[:40]}")

        # Strategies are interleaved inside each repeat so that any drift in
        # machine load affects all of them equally.
        for repeat in range(args.warmup + args.repeats):
            for query in QUERIES:
                for strategy in strategies:
                    report = engine.search(filters=filters, top_k=args.top,
                                           strategy=strategy,
                                           query_vector=query_vectors[query])
                    if repeat >= args.warmup:
                        for stage in STAGES:
                            samples[(label, strategy)][stage].append(
                                report.timings_ms.get(stage, 0.0))
        print(f"measured: {label}", flush=True)

    # --- report ----------------------------------------------------------
    baseline = summarise(samples[("no filters", "mask")]["total_ms"])
    results = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "index": {
            "patents": len(engine.patents),
            "units": len(engine.units),
            "vector_dim": int(engine.vectors.shape[1]),
            "vectors_mb": round(engine.vectors.nbytes / 1e6, 1),
            "model": engine.model_name,
            "index_load_s": round(index_load_s, 2),
            "model_load_s": round(model_load_s, 2),
        },
        "settings": {"queries": len(QUERIES), "repeats": args.repeats,
                     "warmup": args.warmup, "top_k": args.top},
        "query_embedding_ms": embed,
        "identical_results_across_strategies": not mismatches,
        "mismatches": mismatches,
        "scenarios": [],
    }
    for label, filters in SCENARIOS:
        entry = {"scenario": label, **counts[label], "strategies": {}}
        for strategy in (STRATEGIES if filters.active() else ["mask"]):
            name = strategy if filters.active() else "no-filter"
            entry["strategies"][name] = {
                stage: summarise(samples[(label, strategy)][stage]) for stage in STAGES}
        results["scenarios"].append(entry)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    lines = ["# Timing: filters on vs off", ""]
    env, idx = results["environment"], results["index"]
    lines += [
        f"- Machine: {env['platform']}, {env['cpu_count']} logical CPUs, "
        f"Python {env['python']}, NumPy {env['numpy']}",
        f"- Index: {idx['patents']} patents, {idx['units']} units, "
        f"{idx['vector_dim']}-dim vectors ({idx['vectors_mb']} MB), model `{idx['model']}`",
        f"- Start-up (one-off): index load {idx['index_load_s']} s, "
        f"model load {idx['model_load_s']} s",
        f"- {len(QUERIES)} queries x {args.repeats} repeats per cell "
        f"(after {args.warmup} warm-up rounds), top {args.top} results",
        f"- Query embedding (same for every strategy, not included below): "
        f"median {embed['median']:.1f} ms, p95 {embed['p95']:.1f} ms",
        f"- All strategies returned identical results: "
        f"{'yes' if not mismatches else 'NO - see results.json'}",
        "",
        "## Search time after the query is embedded (ms)",
        "",
        "| Scenario | Eligible patents | Eligible units | Share of units "
        "| prefilter median / p95 | mask median / p95 | postfilter (naive) median / p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in results["scenarios"]:
        cells = []
        for strategy in STRATEGIES:
            stats = entry["strategies"].get(strategy)
            cells.append(f"{stats['total_ms']['median']:.2f} / {stats['total_ms']['p95']:.2f}"
                         if stats else "-")
        if "no-filter" in entry["strategies"]:
            stats = entry["strategies"]["no-filter"]["total_ms"]
            cells = ["-", f"{stats['median']:.2f} / {stats['p95']:.2f} (no filter)", "-"]
        lines.append(
            f"| {entry['scenario']} | {entry['eligible_patents']} | {entry['eligible_units']} "
            f"| {entry['share_of_units']:.1%} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += ["", f"Unfiltered baseline: median {baseline['median']:.2f} ms.", "",
              "## Where the time goes (median ms per stage)", "",
              "| Scenario | Strategy | Resolve filters | Score | Rank | Fetch evidence | Total |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for entry in results["scenarios"]:
        for name, stages in entry["strategies"].items():
            lines.append(
                f"| {entry['scenario']} | {name} | {stages['filter_ms']['median']:.3f} "
                f"| {stages['score_ms']['median']:.2f} | {stages['rank_ms']['median']:.2f} "
                f"| {stages['evidence_ms']['median']:.3f} | {stages['total_ms']['median']:.2f} |")
    lines += ["", "Notes: for prefilter, copying out the eligible vectors, scoring and ranking "
              "are interleaved and reported together under Score. For postfilter, the filters "
              "are checked while walking down the ranking, so that cost is under Rank.", ""]

    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"written: {os.path.join(args.out, 'results.md')} and results.json")


if __name__ == "__main__":
    main()