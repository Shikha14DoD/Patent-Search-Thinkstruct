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

from .search import EXCERPT_MARGIN, MAX_EXCERPTS, STRATEGIES, Filters, PatentSearch


def _quote(value: str) -> str:
    """Quote a value for the shell; single quotes when it holds a phrase in quotes."""
    return f"'{value}'" if '"' in value else f'"{value}"'


def as_command(query: str, filters: Filters, args) -> str:
    """The CLI command that runs a suggestion (works in PowerShell and bash)."""
    parts = ["python -m src.cli"]
    if query:
        parts.append(_quote(query))
    for flag, value in (("--classification", filters.classification),
                        ("--exact-title", filters.exact_title),
                        ("--title-keywords", filters.title_keywords),
                        ("--abstract-keywords", filters.abstract_keywords),
                        ("--title-or-abstract-keywords", filters.title_or_abstract_keywords)):
        if value:
            parts.append(f"{flag} {_quote(value)}")
    if filters.match_any:
        parts.append("--match-any")
    if args.top != 10:
        parts.append(f"--top {args.top}")
    return " ".join(parts)


def shorten(text: str, full: bool, width: int) -> str:
    return text if full else textwrap.shorten(text, width=width, placeholder=" …")


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
    parser.add_argument("--excerpts", type=int, default=MAX_EXCERPTS,
                        help="Most matching pieces shown per result")
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

    report = engine.search(args.query, filters, top_k=args.top, strategy=args.strategy,
                           max_excerpts=args.excerpts)

    if report.filter_counts:
        print("Filters:")
        for name, count in report.filter_counts.items():
            print(f"  {name:<40} {count:>5} patents")
        print()

    if not report.results:
        if report.empty_reason:
            print(f"No results: the {report.empty_reason} filter matched nothing.")
        else:
            print("No results: each filter matches some patents, but none matches them all.")
        if report.suggestions:
            print("\nTry instead:")
            for number, suggestion in enumerate(report.suggestions, start=1):
                print(f"  {number}. {suggestion.message}")
                print(f"     {as_command(suggestion.query, suggestion.filters, args)}")
        return

    if not args.query:
        print("No query given: showing the filtered patents, newest first.\n")

    for result in report.results:
        print(f"#{result.rank}  {result.title}")
        score = f" · score {result.score}" if args.query else ""
        print(f"    {result.patent_id} · {result.classification} · "
              f"published {result.publication_date}{score}")
        print(f"    {result.url}")
        if not result.excerpts:
            # Without a query nothing was matched: the abstract is a preview.
            note = f" ({result.excerpt_note})" if result.excerpt_note else ""
            print(f"    {result.excerpt_label}{note}: {shorten(result.excerpt_text, args.full, 240)}")
        for number, excerpt in enumerate(result.excerpts):
            lead = "Matched" if number == 0 else "Also matched"
            note = f"; {excerpt.note}" if excerpt.note else ""
            width = 240 if number == 0 else 160
            print(f"    {lead} {excerpt.label} ({excerpt.score:.3f}{note}): "
                  f"{shorten(excerpt.text, args.full, width)}")
        if result.more_close_matches:
            plural = "s" if result.more_close_matches > 1 else ""
            print(f"    + {result.more_close_matches} more passage{plural} within "
                  f"{EXCERPT_MARGIN:.2f} of the best match")
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