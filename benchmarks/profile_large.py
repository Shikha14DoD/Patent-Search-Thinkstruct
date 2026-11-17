"""Profile Part 1's indexing pipeline on a sample of the large dataset.

    python benchmarks/profile_large.py --data data/patent_data_large

Times every stage of turning raw patent files into a searchable index, using
exactly Part 1's code, on a sample spread evenly across the dataset's files:

  read     load the JSON files
  clean    clean each patent record (claim numbers, paragraphs)
  split    split patents into passages that fit the model, using its tokenizer
  embed    turn every distinct passage into a vector (no cache: all from scratch)
  index    build the filter indexes and the row layout

Embedding is slow, so it is timed on a small sample (--patents); the cheap
stages and the passage statistics use a larger one (--count-patents). Each
stage is projected to the whole dataset, so the slowest stage is the one to
improve. Writes benchmarks/large/profile.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_index import DEFAULT_MODEL, EmbeddingCache, assemble_bundle, embed_units  # noqa: E402
from src.load_data import TokenCounter, build_units, parse_record, patent_files, spread  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile indexing on the large dataset.")
    parser.add_argument("--data", default="data/patent_data_large")
    parser.add_argument("--patents", type=int, default=50, help="Patents to embed")
    parser.add_argument("--count-patents", type=int, default=2000,
                        help="Patents for the cheap stages and passage statistics")
    parser.add_argument("--files", type=int, default=5,
                        help="Files to sample from, spread evenly across the dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="As in Part 1")
    parser.add_argument("--out", default="benchmarks/large/profile.json")
    args = parser.parse_args()

    paths = patent_files(args.data)
    if not paths:
        raise SystemExit(f"No .json files under {args.data}")
    chosen = spread(paths, args.files)
    timings: dict[str, float] = {}
    rng = random.Random(0)

    # read: whole files, since that is what processing the dataset means.
    start = time.perf_counter()
    loaded = []
    for path in chosen:
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        loaded.append(records if isinstance(records, list) else [])
    timings["read"] = time.perf_counter() - start
    records_read = sum(len(records) for records in loaded)
    usable = [r for records in loaded for r in records
              if isinstance(r, dict) and r.get("doc_number")]
    counted = rng.sample(usable, min(args.count_patents, len(usable)))

    # Cheap stages, on the larger sample.
    start = time.perf_counter()
    patents = [parse_record(record, "") for record in counted]
    timings["clean"] = time.perf_counter() - start

    from transformers import AutoTokenizer
    from src.model import MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=MODEL_REVISION)
    start = time.perf_counter()
    units, _ = build_units(patents, TokenCounter(tokenizer))
    timings["split"] = time.perf_counter() - start

    start = time.perf_counter()
    assemble_bundle(patents, units, DEFAULT_MODEL)
    timings["index"] = time.perf_counter() - start

    # Embedding, on the first --patents of them (already a random sample).
    embedded = {p.patent_id for p in patents[:args.patents]}
    embed_texts = [u.text for u in units if u.patent_id in embedded]
    with tempfile.TemporaryDirectory() as empty_cache:
        cache = EmbeddingCache(empty_cache, DEFAULT_MODEL)
        _, embed_stats = embed_units(embed_texts, DEFAULT_MODEL, cache, args.batch_size)
    timings["embed"] = embed_stats["embed_s"]
    model_load_s = embed_stats["model_load_s"]

    import torch
    passages_per_patent = [0] * len(patents)
    position = {p.patent_id: i for i, p in enumerate(patents)}
    for unit in units:
        passages_per_patent[position[unit.patent_id]] += 1
    tokens = [len(tokenizer(t)["input_ids"]) for t in embed_texts]
    n = len(patents)
    n_embedded = len(embedded)
    total_files = len(paths)
    patents_in_dataset = round(records_read / len(chosen) * total_files)

    # Per-patent cost of each stage; reading is per record read, not per sampled patent.
    per_patent = {stage: seconds / n for stage, seconds in timings.items()}
    per_patent["read"] = timings["read"] / records_read
    per_patent["embed"] = timings["embed"] / n_embedded
    total_per_patent = sum(per_patent.values())

    print(f"\nSample: {n} patents ({n_embedded} embedded) from {len(chosen)} of "
          f"{total_files} files "
          f"(~{patents_in_dataset:,} patents in the dataset); "
          f"torch threads: {torch.get_num_threads()}")
    print(f"Passages per patent: mean {statistics.mean(passages_per_patent):.1f}, "
          f"median {statistics.median(passages_per_patent):.0f}, "
          f"90th percentile {percentile(passages_per_patent, 0.9):.0f}, "
          f"max {max(passages_per_patent)}")
    print(f"Tokens per passage: mean {statistics.mean(tokens):.0f}, "
          f"median {statistics.median(tokens):.0f}; "
          f"exact repeats among {len(units)} passages: "
          f"{1 - len({u.text for u in units}) / len(units):.1%}")
    rate = embed_stats["embedded_now"] / timings["embed"] if timings["embed"] else 0
    print(f"Embedding: {rate:.1f} passages/s, "
          f"{sum(tokens) / timings['embed'] if timings['embed'] else 0:.0f} tokens/s "
          f"(batch {args.batch_size}); model load {model_load_s:.1f} s, not counted below\n")

    print(f"| Stage | Sample (s) | Per patent (ms) | Share | Whole dataset (projected) |")
    print("|---|---:|---:|---:|---:|")
    rows = []
    for stage in ("read", "clean", "split", "embed", "index"):
        whole_s = per_patent[stage] * patents_in_dataset
        share = per_patent[stage] / total_per_patent
        human = (f"{whole_s / 86400:.1f} days" if whole_s >= 86400 else
                 f"{whole_s / 3600:.1f} h" if whole_s >= 3600 else f"{whole_s / 60:.1f} min")
        print(f"| {stage} | {timings[stage]:.2f} | {per_patent[stage] * 1000:.1f} | "
              f"{share:.1%} | {human} |")
        rows.append({"stage": stage, "sample_s": round(timings[stage], 3),
                     "per_patent_ms": round(per_patent[stage] * 1000, 2),
                     "share": round(share, 4), "whole_dataset_s": round(whole_s)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({
            "sample_patents": n, "embedded_patents": n_embedded, "sample_files": [os.path.relpath(p) for p in chosen],
            "records_read": records_read, "dataset_files": total_files,
            "dataset_patents_estimate": patents_in_dataset,
            "torch_threads": torch.get_num_threads(), "batch_size": args.batch_size,
            "passages": len(units), "distinct_passages": len({u.text for u in units}),
            "embedded_passages": embed_stats["embedded_now"],
            "passages_per_patent": {"mean": round(statistics.mean(passages_per_patent), 1),
                                    "median": statistics.median(passages_per_patent),
                                    "p90": percentile(passages_per_patent, 0.9),
                                    "max": max(passages_per_patent)},
            "tokens_per_passage": {"mean": round(statistics.mean(tokens), 1),
                                   "median": statistics.median(tokens)},
            "embed_passages_per_s": round(rate, 1), "model_load_s": round(model_load_s, 1),
            "stages": rows}, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
