# Part 3: efficiency on the large dataset

[← Back to the README](../README.md)

The brief: measure how long it takes to process the large dataset (or a portion of it) into the
database and to search it, improve both, and record the improvements.

**Why I picked efficiency and large data.**
- **It is measured, not judged.** Every result is a time or an agreement with exact search, so
  nothing depends on manual grading.
- **It is where a real system breaks first.** Part 1 has about 10^3 patents; about 10^7 are filed
  every year. Before anything else, a deployment has to know how processing and search behave as
  the data grows.
- **It tests Part 2's design.** It replaces Part 2's estimates (passages per patent, embedding
  rate) with measurements on real data; the measured 147 passages per patent now feeds Part 2's
  cost section.

**What "the database" is here.** Part 1's search index: passage vectors plus filter lookup
tables, written by `build_index`. Processing means splitting patents into passages and embedding
them. PostgreSQL only tracks ingestion status in Part 2, so it is not on this path.

**Data.** The large sample is 471 batch files of about 1,000 patents each (about 471,000
patents, 30.7 GB). The index used here takes 300 patents spread evenly across the files, plus
Part 1's 640: **940 patents, 75,018 passages**, 2.2 times Part 1. Loading it surfaced two data
problems, now handled: a stray non-batch file (`sample.json`) that silently dropped patents, and
38% of classifications written as the text "N/A", now treated as missing so no filter matches
them.

All timings are on the same 8-core Windows laptop as Part 1. Identical runs varied by up to 45%
(the laptop slows as it heats), so build comparisons use a warm-up run, alternate the order
(A, B, B, A) and average.

## Processing: measured

| Measure | Result |
|---|---|
| Share of build time spent embedding | **99.5%** (loading and filter tables are the rest) |
| Passages per patent | 147 on average (median 107, 90th percentile 271), against 54 in Part 1 |
| Complete build from scratch, 20 patents (2,903 passages) | 141.8 s, of which 122.9 s embedding (about 23 passages a second) |
| Rebuild of the 75,018-passage index when embeddings are cached | 39 s (25 s loading, 12.7 s HNSW graph) |
| All 471,000 patents on this laptop | about 26 days |
| 10^7 patents a year | 1.47 billion passages a year: 47 a second around the clock, twice this laptop's rate |

## Processing: four attempts, none adopted

Embedding is the only part worth speeding up, so every attempt targeted it, and each was judged
on a complete build with identical results.

| Attempt | Result | Why |
|---|---|---|
| ONNX Runtime instead of PyTorch (plain and optimised) | Slower at every batch size from 8 to 64 | Measured, not assumed: PyTorch was faster on this laptop |
| 8-bit ONNX model | Changed results: 8.4 to 8.6 of each top 10 the same | Rejected: a quality trade with no relevance measurement to justify it |
| Batch 16 instead of 64 | 1.3–1.4× faster on 1,000 passages in random order, but **0.99×** in a real build | The build sorts passages by length, so batches have almost no padding left to save |
| 4 embedding processes, one thread each | **0.85×** in a real build (embedding alone 1.19×, within noise) | Each process loads its own model, and one PyTorch process already uses every core |

So on a CPU, embedding is limited by raw compute, and the defaults stay as in Part 1 (batch 64,
one process). The lever is a GPU: Part 2 costs a one-off embedding of every passage at $31–188 on
spot GPUs ([design, section 8.3](PART2_DESIGN.md)).

## Search: improved

Exact search scores every passage, so its time grows in step with the index. The engine now
builds an **HNSW graph** (FAISS, `M=32`, `efConstruction=128`) automatically for 50,000 vectors
or more, and uses it for searches **without filters**. It fetches candidate passages, widening
until it has 10 patents, then **rescores the top 30 patents exactly**, so scores and evidence
passages match exact search. Searches with any filter keep Part 1's exact path: forcing HNSW
there dropped recall to as low as 60% with a broad filter and 33% with a narrow one. A graph that
no longer matches the index is detected rather than used.

Unfiltered search on 940 patents, 75,018 passages (median of 20 repeats):

| Query set | Retrieval: exact → HNSW (ms) | Complete query (ms) | Recall@10 | Same evidence |
|---|---:|---:|---:|---:|
| Part 1's 12 vehicle queries | 7.06 → 2.61 (**2.7×**) | 42.5 → 29.2 (1.5×) | 100% | 100% |
| 8 queries from other fields | 5.39 → 1.59 (**3.4×**) | 21.4 → 15.8 (1.4×) | 99% (min 90%) | 100% |

The complete query gains less because it includes embedding the query, which HNSW doesn't change.
Filtered searches run at the same speed as before (0.8–1.2×) with 100% recall.

**The gain grows with the index** (unfiltered retrieval only, 12 queries):

| Patents | Passages | Exact (ms) | HNSW (ms) | Speed-up | Recall@10 (min) |
|---:|---:|---:|---:|---:|---:|
| 118 | 8,938 | 0.51 | 0.46 | 1.1× | 100% (100%) |
| 235 | 18,725 | 1.26 | 0.72 | 1.8× | 99% (90%) |
| 470 | 37,562 | 2.61 | 0.79 | 3.3× | 100% (100%) |
| 940 | 75,018 | 4.77 | 0.82 | **5.8×** | 99% (90%) |

Exact time roughly doubles each time the index doubles; HNSW barely moves. That is why Part 2 uses an
approximate vector index in OpenSearch.

## How to run Part 3

Run every command from the repository's top folder.

Set up Part 1 first (install, and ideally build its index, whose embeddings are reused). Then
unzip the large dataset into `data/patent_data_large/`. Its batch files can stay in their
subfolders: every `patents*.json` file below that folder is read, and any other file is ignored.

```
python -m src.build_index --data data/patent_data_small data/patent_data_large:300 --out index_large
python benchmarks/profile_large.py --data data/patent_data_large
python benchmarks/embed_speed.py --data data/patent_data_large
python benchmarks/build_ab.py --data data/patent_data_large:20 --before 64:1 --after 64:4
python benchmarks/search_large.py --index index_large
python benchmarks/search_scaling.py --index index_large
```

`data/patent_data_large:300` takes 300 patents spread evenly across the files, the same sample
every time (fixed seed).

| Command | Time on the laptop | Writes |
|---|---|---|
| `build_index` (large index) | about 40 min from scratch; under a minute when cached | `index_large/` |
| `profile_large.py` | 6–8 min | `benchmarks/large/profile.json` |
| `embed_speed.py` | 6–8 min | `benchmarks/large/embed_speed.json` |
| `build_ab.py` | 12–15 min (5 complete builds) | `benchmarks/large/build_ab_workers.json` |
| `search_large.py` | about 3 min | `benchmarks/large/search.json` |
| `search_scaling.py` | 1–2 min | `benchmarks/large/search_scaling.json` |

Each benchmark overwrites its committed result file, so `git diff benchmarks/large/` shows how a
new run compares. Absolute times depend on the machine; the speed-ups and the recall and
same-evidence checks are what should reproduce. Decisions D21 to D25 in
[`DECISIONS.md`](../DECISIONS.md) explain each choice.
