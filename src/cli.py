"""Command-line search, used for reproducible runs and quick checks.

Examples:
    python -m src.cli "sensor that detects a flat tire"
    python -m src.cli "wheel rim" --classification B60B --top 5
    python -m src.cli --title-keywords hub --classification B60B
    python -m src.cli --exact-title "PNEUMATIC TIRE"
"""

from __future__ import annotations

import argparse
import textwrap
import time

from .search import STRATEGIES, Filters, PatentSearch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search patent claims, abstracts and descriptions.")
    parser.add_argument("query", nargs="?", default="",
                        help="Free-text query; ranks results by meaning")
    parser.add_argument("--index", default="index")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--classification", default="",
                        help='Code prefix, e.g. B60B (several: "B60B,B60C11")')
    parser.add_argument("--title-keywords", default="")
    parser.add_argument("--abstract-keywords", default="")
    parser.add_argument("--title-or-abstract-keywords", default="")
    parser.add_argument("--exact-title", default="")
    parser.add_argument("--match-any", action="store_true",
                        help="Match any keyword instead of all of them")
    parser.add_argument("--strategy", choices=STRATEGIES, default="auto",
                        help="How filters are applied; auto picks by selectivity")
    parser.add_argument("--full", action="store_true", help="Print whole excerpts")
    args = parser.parse_args()

    engine = PatentSearch(args.index)
    filters = Filters(
        classification=args.classification,
        title_keywords=args.title_keywords,
        abstract_keywords=args.abstract_keywords,
        title_or_abstract_keywords=args.title_or_abstract_keywords,
        exact_title=args.exact_title,
        match_any=args.match_any,
    )

    # Loading the model is a one-off start-up cost, so it is timed apart
    # from the search itself.
    model_load_s = 0.0
    if args.query:
        start = time.perf_counter()
        engine.model
        model_load_s = time.perf_counter() - start

    report = engine.search(args.query, filters, top_k=args.top, strategy=args.strategy)

    if report.filter_counts:
        print("Filters:")
        for name, count in report.filter_counts.items():
            print(f"  {name:<40} {count:>5} patents")
        print()

    if not report.results:
        if report.empty_reason:
            print(f"No results: the {report.empty_reason} filter matched nothing.")
            print("Try removing that filter, or put those words in the query instead.")
        else:
            print("No results.")
        return

    if not args.query:
        print("No query given: showing the filtered patents, newest first.\n")

    for result in report.results:
        print(f"#{result.rank}  {result.title}")
        score = f" · score {result.score}" if args.query else ""
        print(f"    {result.patent_id} · {result.classification} · "
              f"published {result.publication_date}{score}")
        print(f"    {result.url}")
        excerpt = result.excerpt_text if args.full else textwrap.shorten(
            result.excerpt_text, width=240, placeholder=" …")
        note = f" ({result.excerpt_note})" if result.excerpt_note else ""
        # Without a query nothing was matched: the abstract is shown as a preview.
        prefix = "Matched " if args.query else ""
        print(f"    {prefix}{result.excerpt_label}{note}: {excerpt}")
        print()

    if args.query:
        print(f"Strategy {report.strategy}: {report.candidate_patents} of "
              f"{len(engine.patents)} patents eligible, "
              f"{report.scored_units} of {report.total_units} units scored")
    else:
        print(f"{report.candidate_patents} of {len(engine.patents)} patents match the filters")
    stages = " · ".join(f"{name.replace('_ms', '')} {value:.1f} ms"
                        for name, value in report.timings_ms.items())
    print(f"Timing: {stages}")
    if model_load_s:
        print(f"(one-off model load: {model_load_s:.1f} s, not part of the search time)")


if __name__ == "__main__":
    main()