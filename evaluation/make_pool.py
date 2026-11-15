"""Build the blind judging sheet for the relevance evaluation.

For each query in queries.json, takes the top 5 patents from the semantic
search and the top 5 from the BM25 keyword baseline, pools them (a patent
found by both appears once), shuffles them, and writes judgments.csv for a
person to fill in. Which method found which patent is written separately to
pool_key.json, so the judge doesn't see it while judging.

Usage:
    python evaluation/make_pool.py
then open evaluation/judgments.csv, fill the "relevance" column
(2 = relevant, 1 = partly relevant, 0 = not relevant) and run
    python evaluation/score.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.bm25 import BM25  # noqa: E402
from src.search import Filters, PatentSearch  # noqa: E402

TOP_K = 5
SEED = 7  # fixed, so the shuffled order is the same on every run


def short(text: str, width: int) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" …")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the blind judging sheet.")
    parser.add_argument("--index", default="index")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite judgments.csv even if it already has judgments")
    args = parser.parse_args()

    sheet = HERE / "judgments.csv"
    if sheet.exists() and not args.force:
        with open(sheet, encoding="utf-8-sig", newline="") as handle:
            if any(row.get("relevance", "").strip() for row in csv.DictReader(handle)):
                sys.exit("judgments.csv already has judgments; use --force to overwrite it.")

    queries = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    engine = PatentSearch(args.index)
    bm25 = BM25(engine)
    rng = random.Random(SEED)

    rows, key = [], {}
    for q in queries:
        filters = Filters(classification=q.get("classification", ""))
        found: dict[str, dict] = {}

        semantic = engine.search(q["query"], filters, top_k=TOP_K)
        for result in semantic.results:
            entry = found.setdefault(result.patent_id, {"methods": {}, "passages": {}})
            entry["methods"]["semantic"] = result.rank
            entry["passages"][result.excerpt_label] = result.excerpt_text

        for rank, (patent_id, _, row) in enumerate(bm25.search(q["query"], filters, TOP_K), 1):
            unit = engine.units[row]
            entry = found.setdefault(patent_id, {"methods": {}, "passages": {}})
            entry["methods"]["bm25"] = rank
            entry["passages"][unit["label"]] = unit["display_text"]

        patent_ids = list(found)
        rng.shuffle(patent_ids)  # hide which method ranked what
        for number, patent_id in enumerate(patent_ids, start=1):
            item = f"{q['id']}-{number}"
            patent = engine.patents[patent_id]
            passages = " | ".join(f"[{label}] {short(text, 350)}"
                                  for label, text in found[patent_id]["passages"].items())
            rows.append({
                "item": item, "query": q["query"], "relevant_if": q["relevant_if"],
                "patent_id": patent_id, "title": patent["title"],
                "abstract": short(patent["abstract"], 500), "matched_passages": passages,
                "relevance": "", "note": "",
            })
            key[item] = {"query_id": q["id"], "patent_id": patent_id,
                         "ranks": found[patent_id]["methods"]}

    fields = ["item", "query", "relevant_if", "patent_id", "title", "abstract",
              "matched_passages", "relevance", "note"]
    # utf-8-sig so Excel opens it with the right characters
    with open(sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "pool_key.json").write_text(json.dumps(
        {"top_k": TOP_K, "model": engine.model_name, "items": key}, indent=2), encoding="utf-8")

    print(f"{len(rows)} items to judge for {len(queries)} queries -> {sheet}")
    print("Fill the relevance column: 2 = relevant, 1 = partly relevant, 0 = not relevant.")
    print("Don't open pool_key.json until you've finished judging.")


if __name__ == "__main__":
    main()