"""Score the relevance evaluation once judgments.csv is filled in.

Reads the judgments (2 = relevant, 1 = partly, 0 = not relevant) and the
hidden key of which method found each patent, then reports for each method:

  P@5 strict   share of its top 5 judged relevant (2)
  P@5 lenient  share of its top 5 judged relevant or partly relevant (2 or 1)
  MRR          1 / rank of its first relevant result (0 if none in the top 5)

Writes evaluation/results.md.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("semantic", "bm25")


def main() -> None:
    queries = {q["id"]: q for q in json.loads((HERE / "queries.json").read_text(encoding="utf-8"))}
    key = json.loads((HERE / "pool_key.json").read_text(encoding="utf-8"))
    top_k = key["top_k"]

    with open(HERE / "judgments.csv", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [r["item"] for r in rows if r["relevance"].strip() not in {"0", "1", "2"}]
    if missing:
        sys.exit(f"{len(missing)} items have no valid relevance (0, 1 or 2), e.g. {missing[:5]}")

    grade = {r["item"]: int(r["relevance"]) for r in rows}
    ranked = defaultdict(dict)  # (query, method) -> {rank: grade}
    for item, info in key["items"].items():
        for method, rank in info["ranks"].items():
            ranked[(info["query_id"], method)][rank] = grade[item]

    def metrics(query_id: str, method: str) -> tuple[float, float, float]:
        grades = ranked.get((query_id, method), {})
        strict = sum(1 for g in grades.values() if g == 2) / top_k
        lenient = sum(1 for g in grades.values() if g >= 1) / top_k
        first = min((rank for rank, g in grades.items() if g == 2), default=None)
        return strict, lenient, (1 / first if first else 0.0)

    lines = ["# Relevance evaluation: semantic search vs BM25 keyword baseline", "",
             f"Top {top_k} patents per method for {len(queries)} queries, pooled, shuffled and "
             "judged blind by one person (2 = relevant, 1 = partly, 0 = not relevant) against a "
             "relevance rule written before any results were seen (`queries.json`).", "",
             "| Query | Type | Semantic P@5 strict / lenient | BM25 P@5 strict / lenient "
             "| Semantic MRR | BM25 MRR |",
             "|---|---|---:|---:|---:|---:|"]
    totals = {m: [0.0, 0.0, 0.0] for m in METHODS}
    for query_id, q in queries.items():
        cells = []
        for method in METHODS:
            s, l, r = metrics(query_id, method)
            totals[method] = [a + b for a, b in zip(totals[method], (s, l, r))]
            cells.append((s, l, r))
        (ss, sl, sr), (bs, bl, br) = cells
        label = q["query"] if len(q["query"]) <= 60 else q["query"][:57] + "…"
        lines.append(f"| {query_id}: {label} | {q['type']} | {ss:.1f} / {sl:.1f} | {bs:.1f} / {bl:.1f} "
                     f"| {sr:.2f} | {br:.2f} |")
    n = len(queries)
    (ss, sl, sr), (bs, bl, br) = ([v / n for v in totals[m]] for m in METHODS)
    lines += [f"| **Mean** | | **{ss:.2f} / {sl:.2f}** | **{bs:.2f} / {bl:.2f}** "
              f"| **{sr:.2f}** | **{br:.2f}** |", "",
              f"{len(rows)} judged items. One judge, {n} queries: this is evidence about these "
              "queries, not a measurement of accuracy over all patents.", ""]

    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()