"""Complete index-build time, before and after, for two build settings.

    python benchmarks/build_ab.py --data data/patent_data_large:20 --before 64:1 --after 64:4

A setting is batch_size:workers (embedding processes, one CPU thread each when
more than one). Defaults compare Part 1's setting (64:1) with 4 processes (64:4).
An earlier run compared batch sizes (--before 64:1 --after 16:1 --out
benchmarks/large/build_ab_batch.json): batch 16 was no faster in a real build.

An untimed warm-up build runs first: the first build after start-up pays one-off
costs (loading libraries and the model from a cold disk cache; 93 s against
12 s in testing), which would otherwise be charged to whichever setting ran
first. Then it builds the same sample twice per setting with the real build command, each time
into a fresh folder with a fresh, empty embedding cache, so nothing is reused
and both settings do exactly the same work. Runs in the order before, after,
after, before, so a laptop that slows down (or speeds up) during the test
affects both settings equally. Reports the complete build time (loading,
cleaning, splitting, model load, embedding, filter indexes, writing) and the
embedding time alone, and checks that both settings produce the same vectors.
Writes benchmarks/large/build_ab_workers.json by default (change with --out).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_setting(text: str) -> tuple[int, int]:
    batch, _, workers = text.partition(":")
    return int(batch), int(workers or 1)


def setting_name(setting: tuple[int, int]) -> str:
    batch, workers = setting
    return f"batch {batch}, {workers} process{'es' if workers > 1 else ''}"


def build(data: list[str], setting: tuple[int, int], workdir: str) -> dict:
    batch, workers = setting
    out, cache = os.path.join(workdir, "index"), os.path.join(workdir, "cache")
    command = [sys.executable, "-m", "src.build_index", "--data", *data, "--out", out,
               "--cache", cache, "--batch-size", str(batch), "--workers", str(workers),
               "--hnsw", "no"]
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    wall_s = time.perf_counter() - start
    with open(os.path.join(out, "build_stats.json"), encoding="utf-8") as handle:
        stats = json.load(handle)
    return {"setting": setting_name(setting), "wall_s": round(wall_s, 1),
            "embed_s": stats["timings_s"]["embed_s"],
            "embedded": stats["embedding"]["embedded_now"],
            "units": stats["units"], "patents": stats["patents"],
            "vectors": np.load(os.path.join(out, "vectors.npy"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index-build time for two settings.")
    parser.add_argument("--data", nargs="+", default=["data/patent_data_large:20"])
    parser.add_argument("--before", default="64:1", help="batch:workers (Part 1: 64:1)")
    parser.add_argument("--after", default="64:4", help="batch:workers")
    parser.add_argument("--out", default="benchmarks/large/build_ab_workers.json")
    args = parser.parse_args()
    before_setting, after_setting = parse_setting(args.before), parse_setting(args.after)

    with tempfile.TemporaryDirectory() as workdir:
        warm = build(args.data, after_setting, workdir)
    print(f"  warm-up build (not counted): {warm['wall_s']:.1f} s", flush=True)

    runs = []
    for setting in (before_setting, after_setting, after_setting, before_setting):
        with tempfile.TemporaryDirectory() as workdir:
            run = build(args.data, setting, workdir)
        runs.append(run)
        print(f"  {run['setting']}: build {run['wall_s']:.1f} s, "
              f"embedding {run['embed_s']:.1f} s ({run['embedded']} passages)", flush=True)

    reference = runs[0]["vectors"]
    cosine = min(float((run["vectors"] * reference).sum(axis=1).min()) for run in runs)
    summary = {}
    for setting in (before_setting, after_setting):
        mine = [r for r in runs if r["setting"] == setting_name(setting)]
        summary[setting_name(setting)] = {
            "build_s": round(statistics.mean(r["wall_s"] for r in mine), 1),
            "embed_s": round(statistics.mean(r["embed_s"] for r in mine), 1)}
    before, after = summary[setting_name(before_setting)], summary[setting_name(after_setting)]
    print(f"\nSample: {runs[0]['patents']} patents, {runs[0]['units']} passages, "
          f"{runs[0]['embedded']} embedded from scratch in every run. "
          f"Lowest cosine between any run's vectors and the first: {cosine:.4f}\n")
    print(f"| Measure | Before: {setting_name(before_setting)} | After: {setting_name(after_setting)} | "
          "Speed-up |")
    print("|---|---:|---:|---:|")
    for label, key in (("Complete build (s)", "build_s"), ("Embedding only (s)", "embed_s")):
        print(f"| {label} | {before[key]} | {after[key]} | {before[key] / after[key]:.2f}× |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"data": args.data, "patents": runs[0]["patents"], "units": runs[0]["units"],
                   "embedded_per_run": runs[0]["embedded"],
                   "order": [r["setting"] for r in runs],
                   "warm_up_s": warm["wall_s"],
                   "runs": [{k: v for k, v in r.items() if k != "vectors"} for r in runs],
                   "mean": summary, "min_cosine_vs_first_run": round(cosine, 5)},
                  handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()