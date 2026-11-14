"""Build the search index: one pass over the data, saved to disk.

Everything slow happens here, once:
  - embedding every unit with a sentence-transformer model
  - building the inverted indexes used by the filters

Searching then only has to embed the query and do lookups, which is why a
search takes milliseconds while this script takes minutes.

Embeddings are cached on disk, keyed by (model, exact text). A rebuild only
embeds text it has not seen before, so changing a cleaning rule costs
seconds instead of the full embedding time, and an interrupted build
resumes where it stopped.

Usage:
    python -m src.build_index --data data/patent_data_small --out index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict

import numpy as np

from .load_data import load_corpus
from .text_utils import normalize_title, tokenize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Texts embedded between two cache writes. Smaller = less work lost if the
# build is interrupted.
CHECKPOINT_TEXTS = 2048


# -- embedding cache --------------------------------------------------------
def text_key(model_name: str, text: str) -> str:
    """Cache key: the model and the exact text decide the vector."""
    return hashlib.sha1(f"{model_name}\x00{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Append-only cache: keys.txt (one hash per line) + vectors.f32 (raw rows)."""

    def __init__(self, directory: str, model_name: str):
        slug = model_name.replace("/", "__")
        self.directory = os.path.join(directory, slug)
        os.makedirs(self.directory, exist_ok=True)
        self.keys_path = os.path.join(self.directory, "keys.txt")
        self.vectors_path = os.path.join(self.directory, "vectors.f32")
        self.dim_path = os.path.join(self.directory, "dim.txt")
        self.rows: dict[str, int] = {}
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not (os.path.exists(self.keys_path) and os.path.exists(self.dim_path)):
            return
        with open(self.dim_path, encoding="utf-8") as handle:
            dim = int(handle.read().strip())
        with open(self.keys_path, encoding="utf-8") as handle:
            keys = [line.strip() for line in handle if line.strip()]
        vectors = np.fromfile(self.vectors_path, dtype=np.float32)
        complete = min(len(keys), vectors.size // dim)  # tolerate a cut-off write
        self.vectors = vectors[: complete * dim].reshape(complete, dim)
        self.rows = {key: row for row, key in enumerate(keys[:complete])}

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, key: str) -> np.ndarray | None:
        row = self.rows.get(key)
        return None if row is None else self.vectors[row]

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if not os.path.exists(self.dim_path):
            with open(self.dim_path, "w", encoding="utf-8") as handle:
                handle.write(str(vectors.shape[1]))
        with open(self.vectors_path, "ab") as handle:
            handle.write(vectors.tobytes())
        with open(self.keys_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(keys) + "\n")
        start = len(self.rows)
        self.vectors = (vectors if self.vectors.size == 0
                        else np.vstack([self.vectors, vectors]))
        for offset, key in enumerate(keys):
            self.rows[key] = start + offset

    def seed_from_index(self, index_dir: str, model_name: str) -> int:
        """Reuse vectors from an index built before the cache existed."""
        pkl, npy = os.path.join(index_dir, "index.pkl"), os.path.join(index_dir, "vectors.npy")
        if len(self) or not (os.path.exists(pkl) and os.path.exists(npy)):
            return 0
        with open(pkl, "rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("model_name") != model_name:
            return 0
        vectors = np.load(npy)
        keys, rows, seen = [], [], set()
        for row, unit in enumerate(bundle["units"]):
            key = text_key(model_name, unit["text"])
            if key not in seen:
                seen.add(key)
                keys.append(key)
                rows.append(row)
        self.add(keys, vectors[rows])
        return len(keys)


def embed_units(texts: list[str], model_name: str, cache: EmbeddingCache,
                batch_size: int) -> tuple[np.ndarray, dict]:
    """Return one vector per text, embedding only what the cache lacks."""
    keys = [text_key(model_name, text) for text in texts]
    missing: dict[str, str] = {}
    for key, text in zip(keys, texts):
        if key not in cache.rows and key not in missing:
            missing[key] = text
    stats = {"texts": len(texts), "distinct_texts": len(set(keys)),
             "cache_hits": len(set(keys)) - len(missing), "embedded_now": len(missing),
             "model_load_s": 0.0, "embed_s": 0.0}

    if missing:
        from sentence_transformers import SentenceTransformer  # imported late: slow

        start = time.perf_counter()
        model = SentenceTransformer(model_name)
        stats["model_load_s"] = time.perf_counter() - start
        print(f"model loaded in {stats['model_load_s']:.1f}s; "
              f"embedding {len(missing)} new texts "
              f"({stats['cache_hits']} already cached)")

        # Longest first keeps similar lengths together, which wastes less
        # padding, and makes the slowest part of the run show up first.
        todo = sorted(missing.items(), key=lambda item: -len(item[1]))
        start = time.perf_counter()
        for begin in range(0, len(todo), CHECKPOINT_TEXTS):
            part = todo[begin:begin + CHECKPOINT_TEXTS]
            vectors = model.encode(
                [text for _, text in part],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine similarity becomes a dot product
                show_progress_bar=False,
            ).astype(np.float32)
            cache.add([key for key, _ in part], vectors)
            done = begin + len(part)
            elapsed = time.perf_counter() - start
            print(f"  embedded {done}/{len(todo)} "
                  f"({done / elapsed:.0f} texts/s, {elapsed:.0f}s elapsed)", flush=True)
        stats["embed_s"] = time.perf_counter() - start
    else:
        print(f"all {stats['distinct_texts']} distinct texts found in the cache")

    return np.vstack([cache.get(key) for key in keys]).astype(np.float32), stats


# -- filter indexes ---------------------------------------------------------
def build_inverted_index(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Map each normalised word to the patents whose field contains it."""
    index: dict[str, set[str]] = {}
    for patent_id, text in pairs:
        for token in set(tokenize(text)):
            index.setdefault(token, set()).add(patent_id)
    return index


def build_phrase_index(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Normalised field text per patent, for phrase ("quoted") matching."""
    return {patent_id: " ".join(tokenize(text)) for patent_id, text in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the patent search index.")
    parser.add_argument("--data", default="data/patent_data_small",
                        help="Folder holding the patents_ipa*.json files")
    parser.add_argument("--out", default="index", help="Folder to write the index into")
    parser.add_argument("--cache", default="embedding_cache",
                        help="Folder for the embedding cache")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    patents, units, ingest_report = load_corpus(args.data)
    timings["load_and_clean_s"] = time.perf_counter() - start
    print(f"loaded {len(patents)} patents -> {len(units)} units "
          f"in {timings['load_and_clean_s']:.1f}s")
    for key, value in ingest_report.items():
        print(f"    {key:<28} {value}")

    # --- embeddings -------------------------------------------------------
    cache = EmbeddingCache(args.cache, args.model)
    seeded = cache.seed_from_index(args.out, args.model)
    if seeded:
        print(f"seeded the embedding cache with {seeded} vectors from the existing index")

    vectors, embed_stats = embed_units([unit.text for unit in units], args.model,
                                       cache, args.batch_size)
    timings["model_load_s"] = embed_stats.pop("model_load_s")
    timings["embed_s"] = embed_stats.pop("embed_s")
    print(f"{len(units)} unit vectors ready ({vectors.nbytes / 1e6:.0f} MB)")

    # --- filter indexes ---------------------------------------------------
    start = time.perf_counter()
    titles = [(p.patent_id, p.title) for p in patents]
    abstracts = [(p.patent_id, p.abstract) for p in patents]

    title_index = build_inverted_index(titles)
    abstract_index = build_inverted_index(abstracts)
    # "keywords in the title or abstract" as one box, per the brief
    title_or_abstract_index: dict[str, set[str]] = {}
    for source in (title_index, abstract_index):
        for token, ids in source.items():
            title_or_abstract_index.setdefault(token, set()).update(ids)

    exact_title_index: dict[str, list[str]] = {}
    for patent_id, title in titles:
        exact_title_index.setdefault(normalize_title(title), []).append(patent_id)

    # Classification is matched by prefix, so every prefix of every code gets
    # its own entry: a "B60B" lookup is then a dict hit, not a scan. The codes
    # come from the data, so nothing here is specific to vehicle patents.
    classification_prefix: dict[str, set[str]] = {}
    for patent in patents:
        code = patent.classification.upper()
        for length in range(1, len(code) + 1):
            classification_prefix.setdefault(code[:length], set()).add(patent.patent_id)

    timings["filter_indexes_s"] = time.perf_counter() - start
    print(f"filter indexes built in {timings['filter_indexes_s']:.1f}s")

    # --- unit layout ------------------------------------------------------
    # Units are generated patent by patent, so each patent owns a contiguous
    # block of rows. Storing the block bounds lets the search work on whole
    # blocks instead of looking at units one by one.
    unit_rows: dict[str, tuple[int, int]] = {}
    for position, unit in enumerate(units):
        if unit.patent_id not in unit_rows:
            unit_rows[unit.patent_id] = (position, position + 1)
        else:
            first, _ = unit_rows[unit.patent_id]
            unit_rows[unit.patent_id] = (first, position + 1)

    # Claim objects are dropped from the stored patent record: the claim text
    # already lives in the units, and keeping both would double the index.
    patent_records = {}
    for patent in patents:
        record = asdict(patent)
        for key in ("claims", "raw_paragraphs", "paragraphs", "paragraph_numbers"):
            record.pop(key, None)
        record["claim_count"] = len(patent.claims)
        patent_records[patent.patent_id] = record

    bundle = {
        "model_name": args.model,
        "patents": patent_records,
        "units": [asdict(u) for u in units],
        "unit_rows": unit_rows,
        "title_index": title_index,
        "abstract_index": abstract_index,
        "title_or_abstract_index": title_or_abstract_index,
        "title_phrases": build_phrase_index(titles),
        "abstract_phrases": build_phrase_index(abstracts),
        "exact_title_index": exact_title_index,
        "classification_prefix": classification_prefix,
        "ingest_report": ingest_report,
        "timings": timings,
    }

    np.save(os.path.join(args.out, "vectors.npy"), vectors)
    with open(os.path.join(args.out, "index.pkl"), "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    index_bytes = os.path.getsize(os.path.join(args.out, "index.pkl"))
    stats = {
        "patents": len(patents),
        "units": len(units),
        "units_by_kind": {
            kind: sum(1 for u in units if u.kind == kind)
            for kind in ("abstract", "claim", "paragraph")
        },
        "patents_without_description": sum(1 for p in patents if not p.paragraphs),
        "ingest_report": ingest_report,
        "embedding": embed_stats,
        "vector_mb": round(vectors.nbytes / 1e6, 1),
        "metadata_mb": round(index_bytes / 1e6, 1),
        "distinct_title_words": len(title_index),
        "distinct_abstract_words": len(abstract_index),
        "model": args.model,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
    }
    if timings["embed_s"] > 0 and embed_stats["embedded_now"]:
        stats["embedding"]["texts_per_second"] = round(
            embed_stats["embedded_now"] / timings["embed_s"], 1)
        # Keep the cost of a from-scratch embedding run on record, because a
        # later cached rebuild would otherwise overwrite it with ~0 seconds.
        with open(os.path.join(args.out, "embedding_run.json"), "w", encoding="utf-8") as handle:
            json.dump({"embedded_texts": embed_stats["embedded_now"],
                       "embed_s": round(timings["embed_s"], 1),
                       "texts_per_second": stats["embedding"]["texts_per_second"],
                       "model": args.model}, handle, indent=2)
    with open(os.path.join(args.out, "build_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"\nindex written to {args.out}/  (total {sum(timings.values()):.1f}s)")


if __name__ == "__main__":
    main()