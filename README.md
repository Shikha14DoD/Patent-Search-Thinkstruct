# Patent search engine

Search 640 vehicle-related US patent applications by meaning, narrow the results with strict
filters, and see exactly which claim or paragraph matched. Built for the Thinkstruct take-home.

Every design choice, with the measurements behind it, is in [`DECISIONS.md`](DECISIONS.md).

## Problem statement

Inventors, attorneys and patent examiners search existing patents to judge whether an idea is
new, to avoid infringing someone else's claims, and to see what is being filed in a field. Two
things make this hard:

- **The same idea is written in different words.** A keyword search for "flat tire sensor"
  misses a claim about "detecting loss of inflation pressure".
- **A whole patent is not evidence.** A typical patent here runs to over 4,000 tokens. A
  researcher needs to know *which claim or paragraph* overlaps with their idea, to check it.

The problem this project solves: **given a description of an idea or a claim, return the most
similar patents, show the specific passages that match, and let the user narrow the results
with strict filters on classification, title and abstract keywords, and exact title.**

It is a retrieval tool for a person to review, not a judge of novelty or infringement.

## What it does

**Input:** a free-text query (a plain-English description, or a pasted claim), plus any of these
optional filters:

| Filter | Example | Meaning |
|---|---|---|
| Classification prefix | `--classification B60B` | Code starts with B60B (vehicle wheels); several codes allowed |
| Title keywords | `--title-keywords hub` | Title contains the word (sensor/sensors and tyre/tire count as the same word) |
| Abstract keywords | `--abstract-keywords "tread groove"` | Abstract contains every word (`--match-any` for any word) |
| Title or abstract keywords | `--title-or-abstract-keywords sensor` | Either field contains the word |
| Exact title | `--exact-title "PNEUMATIC TIRE"` | Title matches, ignoring case and punctuation |

Quoted words inside a filter (`'"wheel hub"'`) must appear together as a phrase. Filters
combine with AND, and they are strict: a patent that fails one never appears.

**Output,** for each result: rank, score, patent number, title, classification, publication
date, a Google Patents link, and up to three matching passages labelled with where they come
from:

```
#3  TIRE WITH SEALANT AND TIRE PUNCTURE SENSOR SYSTEM
    20250108667 · B60C1912FI · published 2025-04-03 · score 0.6099
    https://patents.google.com/patent/US20250108667A1
    Matched Claim 14 (0.610; with Claim 13 as context): The tire according to claim 13 , wherein the sensor system further comprises a transmitter …
    Also matched Claim 17 (0.604; with Claim 15 as context): The tire according to claim 15 , wherein the sensor system comprises a sensor transmitter unit …
    Also matched Claim 18 (0.600; with Claim 17 as context): The tire according to claim 17 , wherein the tire further comprises an elastomer …
    + 17 more passages within 0.10 of the best match
```

**When filters leave nothing,** the engine says which filter matched nothing and suggests
changes that would return patents, each with its count and a command to run it: likely
spellings, near titles, classification codes that exist, "any word" instead of "all words",
searching the words by meaning, or dropping one filter.

## How it works

1. **Load and clean** (`src/load_data.py`). Each patent is split into searchable pieces: its
   abstract, each claim, and each description paragraph. 640 patents become 34,551 pieces,
   about 54 per patent. Cleaning fixes problems measured in the data: claim numbers rebuilt
   after cancelled claims, numerals glued to words separated, fragments under five words
   dropped, and every piece fitted to the model's 256-token input limit.
2. **Build the index** (`src/build_index.py`). Each piece is turned into a 384-number vector
   with `sentence-transformers/all-MiniLM-L6-v2`, so pieces with similar meaning get similar
   vectors. Lookup tables are built for the filters: for each word, which patents contain it in
   their title or abstract; for each classification prefix, which patents have it.
3. **Search** (`src/search.py`). The filters are resolved through the lookup tables, the query
   is embedded once and compared with the eligible pieces, each patent is ranked by its best
   piece, and the best pieces are shown as evidence.

Why pieces rather than whole patents: the model reads 256 tokens and a patent with a
description has a median of 4,195, so whole-patent vectors would judge each patent by its
first page. Pieces make every claim searchable and let the result show exactly what matched
(decision D8).

### Missing and malformed data

No field is missing outright, but 20,819 of 42,040 description paragraphs are empty strings
and 119 patents have no description text at all. **Every patent is kept;** blank paragraphs are
skipped, and patents without a description stay searchable through their title, abstract and
claims (D1). Other data problems found and fixed are listed in `DECISIONS.md` (D2 to D7). The
weekly date in the file names is the A1 **publication** date, not the filing date (D3).

## Hybrid search and efficiency

In the brief, hybrid search means combining similarity search with strict constraints. The
benchmark runs 12 fixed queries, 30 times each, under 8 filter scenarios, and compares three
ways to apply the filters:

- **Prefilter:** find the eligible patents through the lookup tables, copy out only their
  vectors, and score those.
- **Mask:** find the eligible patents the same way, score every vector in place, and hide the
  rest.
- **Postfilter (naive):** rank every vector, then walk down the ranking checking each patent's
  text against the filters, with no lookup tables.

Median milliseconds per search, once the query is embedded (8-core Windows laptop, plugged in
and idle; full tables in [`benchmarks/results.md`](benchmarks/results.md)):

| Filter | Patents left | Share of pieces | Prefilter | Mask | Postfilter (naive) |
|---|---:|---:|---:|---:|---:|
| None | 640 | 100% | – | 2.48 | – |
| Classification B60B | 298 | 44.2% | 9.14 | **2.53** | 5.99 |
| Title/abstract: tread | 156 | 28.9% | 6.22 | **2.56** | 7.33 |
| Classification B60C11 | 101 | 19.1% | 4.32 | **2.42** | 5.88 |
| B60C + title: pneumatic | 94 | 17.7% | 5.39 | **3.21** | 9.54 |
| Abstract: tread groove | 45 | 9.5% | 2.83 | **2.59** | 13.88 |
| Exact title "PNEUMATIC TIRE" | 33 | 6.4% | **2.88** | 3.07 | 10.32 |
| Title/abstract: sensor | 32 | 5.2% | **2.10** | 2.71 | 21.22 |

Bold marks the method the engine picks automatically. All three methods returned identical
results in every case. Embedding the query takes a median of 8.1 ms on top of these times.

**What the numbers show**

- **Naive filtering is 2 to 9 times slower than no filter,** and slowest when the filter is
  selective, because it walks far down the ranking reading patent text before it finds ten
  patents that pass.
- **Resolving the filters through lookup tables costs 0.005 to 0.15 ms.**
- **Filtering first only wins when few pieces survive.** Prefilter beat mask at 5–6% of
  pieces and lost from 9.5% upwards: copying thousands of scattered vectors costs more than
  one multi-threaded multiplication over the whole contiguous matrix. So the engine chooses:
  prefilter at or below 8% of pieces surviving, mask above. With that choice, a broad filter
  costs about the same as no filter (2.4 to 3.2 ms against 2.48 ms in this run; an earlier run
  on the same laptop measured 2.1 to 2.2 ms against 2.26 ms), and a very selective one is
  faster (2.10 ms at 5% of pieces).
- **At this size, the query model is the bottleneck,** not the search: embedding takes about
  three times as long as the whole filtered search.

### What I used to make it efficient

- **Lookup tables built once,** so a filter is a few set operations instead of reading text.
- **Each patent's pieces stored as one contiguous block,** so the eligible pieces are found by
  slicing, not by scanning all 34,551.
- **The filtering method chosen by selectivity,** with the 8% threshold taken from measurements
  on two machines (it moved between them, so it is a tuned value, D12).
- **Exact search as one matrix multiplication** on normalised vectors: at about 2 ms, an
  approximate index would add complexity and lose recall for no visible gain (D9).
- **An embedding cache** keyed by model and exact text: after a cleaning change, a rebuild
  re-embeds only the changed pieces, about 5 minutes instead of 20 (D10).

### What I would use at large scale

At 10 million patents, the same design means about 540 million vectors, roughly 830 GB as
384-dimensional float32. Scoring grows linearly, about 54 ms per million vectors on this
laptop, so exact search would take tens of seconds per query. What would change:

- **Approximate nearest-neighbour search** (HNSW or IVF with product quantization) with
  filter-aware search. Compressing each vector to a 64-byte code would bring the 830 GB to
  about 35 GB.
- **An index partitioned by classification,** so a classification filter skips whole
  partitions instead of copying vectors. That turns filtering into a real speed-up, which it
  can't be at this size.
- **Keyword filters in a search engine or database index** (OpenSearch, or Postgres with GIN
  indexes) and metadata in a database, instead of in-memory dictionaries and a 44 MB pickle
  loaded at start-up.
- **Faster embedding.** The laptop embeds 20 to 29 pieces per second, so embedding 540 million
  pieces this way would take months. Ingestion needs GPUs and parallel workers, and only new
  patents should be embedded each week.
- **A warm query service** that keeps the model loaded and caches embeddings of repeated
  queries, since query embedding dominates latency already.

## How to run Part 1

### 1. Install

Requires **Python 3.12 or newer** (the pinned NumPy 2.5 needs it) and about 3 GB of disk for
PyTorch. It has been run on Python 3.12 (Windows 11) and 3.13 (Linux). The first run downloads
the model (about 90 MB) from Hugging Face.

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Linux, plain pip installs the CUDA build of PyTorch (several GB). To keep it small, run
`python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` before
installing the requirements.

A warning about unauthenticated requests to the Hugging Face Hub is harmless.

### 2. Add the data

Unzip the dataset so the weekly JSON files sit directly in `data/patent_data_small/`:

```
data/patent_data_small/patents_ipa240215.json
data/patent_data_small/patents_ipa240222.json
...   (64 files)
```

### 3. Build the index

```
python -m src.build_index --data data/patent_data_small --out index
```

About 20 minutes on a laptop CPU, almost all of it embedding. It prints an ingest report and
should end with `units longer than the model's 256-token limit: 0` and
`index written to index/`. Expected counts: 640 patents, 34,551 units. Vectors are cached in
`embedding_cache/`, so a rebuild only embeds text it hasn't seen.

### 4. Search

```
python -m src.cli "sensor that detects a flat tire while driving"
python -m src.cli "wheel rim" --classification B60B --top 5
python -m src.cli "tire pressure" --title-or-abstract-keywords sensor
python -m src.cli --exact-title "PNEUMATIC TIRE"
python -m src.cli "tire pressure" --title-keywords "flat tyre warning"
```

The last one matches nothing and shows the suggestions. With no query, filtered patents are
listed newest first. `--excerpts N` changes how many passages each result shows, `--full`
prints them whole, and `python -m src.cli --help` lists every option.

### 5. Reproduce the timings

```
python benchmarks/timing.py
```

A few minutes. It writes `benchmarks/results.md` and `benchmarks/results.json`, and checks first
that all three filtering methods return identical results. Absolute times depend on the
machine; run it plugged in and idle (a run with the laptop busy doubled every stage). The
pattern should hold: naive filtering slowest, prefilter winning only for very selective
filters.

### 6. Run the tests

```
python -m pytest tests -q
```

25 tests check the cleaning rules (claim numbering, numeral separation, token limits) and the
search (excerpts, filter strategies, suggestions) on a tiny hand-made index. They need neither
the model nor the dataset.

## Repository layout

```
src/load_data.py      load, clean and split patents into pieces
src/text_utils.py     word normalisation shared by indexing and filters
src/build_index.py    embeddings, embedding cache, filter lookup tables
src/search.py         search, filters, filtering strategies, excerpts, suggestions
src/cli.py            command-line interface
benchmarks/timing.py  filters-on vs filters-off benchmark
tests/                unit tests
DECISIONS.md          decision log with evidence
```

## Limitations

- **Search quality is not measured yet, only speed.** Results were checked by hand; there is no
  relevance metric, so the model and ranking choices are not yet validated against
  alternatives.
- Keyword filters match words, not synonyms; "search by meaning" covers that, across all text.
- Patents are ranked by their single best passage; a patent with many matching passages shows
  a count but doesn't rank higher (D16).
- Planned next: finding patents similar to a given patent ID. Deferred: keyword ranking
  blended with semantic ranking (BM25), fuzzy title matching, links that open the patent at the
  matching passage, and a "related by meaning" section after strict results (D15).

**What the data can support.** The sample is 640 vehicle-related US patent applications
published between February 2024 and May 2025. It cannot support real prior-art, novelty or
infringement conclusions, which need older patents, other countries, granted patents and
non-patent literature. The engine is not specific to vehicles: classification codes and
vocabulary come from the data.