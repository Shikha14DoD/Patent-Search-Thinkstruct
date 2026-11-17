"""Compare ways of embedding the same passages: speed and agreement with Part 1.

    python benchmarks/embed_speed.py --data data/patent_data_large

Embeds one fixed sample of passages from the large dataset with each variant:

  torch-b16/b32/b64/b128  Part 1's PyTorch model at different batch sizes (b64 = Part 1)
  onnx-fp32            the same model run with ONNX Runtime, full precision
  onnx-optimised       ONNX graph-optimised export, full precision
  onnx-int8            ONNX with 8-bit weights
  (ONNX variants run at batch 64; add -b16 or -b32, e.g. onnx-fp32-b16, for other sizes)

    python benchmarks/embed_speed.py --variants torch-b16 onnx-fp32-b16 onnx-int8-b16

Speed is passages per second (model load not counted). Correctness is checked
against Part 1's vectors (torch-b64): the cosine similarity of each passage's
two vectors, and how many of the top 10 passages for Part 1's 12 benchmark
queries stay the same. Writes benchmarks/large/embed_speed.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.timing import QUERIES  # noqa: E402
from src.build_index import DEFAULT_MODEL  # noqa: E402
from src.model import MODEL_REVISION  # noqa: E402
from src.fast_embed import ONNX_FILES, OnnxEncoder  # noqa: E402
from src.load_data import TokenCounter, build_units, parse_record, patent_files, spread  # noqa: E402


def sample_passages(data: str, count: int, files: int, seed: int = 0) -> list[str]:
    paths = patent_files(data)
    records = []
    for path in spread(paths, files):
        with open(path, encoding="utf-8") as handle:
            records += [r for r in json.load(handle) if isinstance(r, dict) and r.get("doc_number")]
    rng = random.Random(seed)
    from transformers import AutoTokenizer
    counter = TokenCounter(AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=MODEL_REVISION))
    patents = [parse_record(r, "") for r in rng.sample(records, min(200, len(records)))]
    texts = sorted({u.text for u in build_units(patents, counter)[0]})
    return rng.sample(texts, min(count, len(texts)))


def top10(queries: np.ndarray, passages: np.ndarray) -> list[set[int]]:
    return [set(np.argsort(-(passages @ q))[:10]) for q in queries]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding speed and agreement.")
    parser.add_argument("--data", default="data/patent_data_large")
    parser.add_argument("--passages", type=int, default=1000)
    parser.add_argument("--files", type=int, default=5)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Run only these (Part 1's torch-b64 always runs first, as the reference)")
    parser.add_argument("--out", default="benchmarks/large/embed_speed.json")
    args = parser.parse_args()
    if args.variants and args.out == "benchmarks/large/embed_speed.json":
        args.out = "benchmarks/large/embed_speed_extra.json"   # keep the full run's file

    texts = sample_passages(args.data, args.passages, args.files)
    print(f"{len(texts)} distinct passages sampled from {args.files} files")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DEFAULT_MODEL, revision=MODEL_REVISION)

    def torch_encoder(batch):
        return lambda items: model.encode(items, batch_size=batch, convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=False)

    names = args.variants or ["torch-b16", "torch-b32", "torch-b128", *ONNX_FILES]
    variants, encoders = [("torch-b64 (Part 1)", torch_encoder(64))], {}
    for name in names:
        kind, _, batch = name.rpartition("-b")
        if name.startswith("torch-b"):
            variants.append((name, torch_encoder(int(name[7:]))))
            continue
        kind, batch = (kind, int(batch)) if batch.isdigit() else (name, 64)
        if kind not in ONNX_FILES:
            raise SystemExit(f"unknown variant {name!r}")
        encoder = encoders.setdefault(kind, OnnxEncoder(ONNX_FILES[kind]))
        variants.append((name, lambda items, e=encoder, b=batch: e.encode(items, batch_size=b)))

    results, reference, reference_top = [], None, None
    for name, encode in variants:
        encode(texts[:16])                         # warm-up, not timed
        start = time.perf_counter()
        vectors = np.asarray(encode(texts), dtype=np.float32)
        seconds = time.perf_counter() - start
        queries = np.asarray(encode(QUERIES), dtype=np.float32)
        if reference is None:
            reference, reference_top = vectors, top10(queries, vectors)
        cosine = (vectors * reference).sum(axis=1)
        overlap = np.mean([len(a & b) for a, b in zip(top10(queries, vectors), reference_top)])
        results.append({"variant": name, "seconds": round(seconds, 1),
                        "passages_per_s": round(len(texts) / seconds, 1),
                        "cosine_mean": round(float(cosine.mean()), 4),
                        "cosine_min": round(float(cosine.min()), 4),
                        "top10_same": round(float(overlap), 1)})
        print(f"  {name:<20} {len(texts) / seconds:6.1f} passages/s", flush=True)

    base = results[0]["passages_per_s"]
    print("\n| Variant | Passages/s | Speed-up | Cosine vs Part 1 (mean / min) | Same top 10 |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        r["speedup"] = round(r["passages_per_s"] / base, 2)
        print(f"| {r['variant']} | {r['passages_per_s']} | {r['speedup']}× | "
              f"{r['cosine_mean']:.4f} / {r['cosine_min']:.4f} | {r['top10_same']} of 10 |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"passages": len(texts), "cpu_count": os.cpu_count(), "results": results},
                  handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
