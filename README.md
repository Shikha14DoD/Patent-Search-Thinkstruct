# Patent Evidence Explorer

Semantic patent search that shows the claim or paragraph behind every result, with strict
filters on classification, title and abstract.

**Current corpus:** 640 US patent application publications from a vehicle-domain sample
(wheels, tires and couplings), published February 2024 to May 2025. Nothing in the engine is
specific to vehicles: classification codes and vocabulary come from the data.

Built as a take-home project for Thinkstruct. Every design choice, with the evidence behind
it, is in [`DECISIONS.md`](DECISIONS.md). The design for 10 million patents and its proof of
concept are in [Part 2](#part-2-10-million-patents).

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

## Engineering highlights

| Problem found | What the engine does | Evidence |
|---|---|---|
| A third of claims lost their printed number, and cancelled claims shift the numbering | Rebuilds claim numbers from printed numbers and cancelled ranges; flags inferred ones | 96.7% of "according to claim N" references resolve; unit tests |
| The model reads 256 tokens; 17% of dependent claims were losing their own text | Splits text by the model's own token count; a claim's own words always fit | 0 of 34,551 units over the limit; unit tests |
| Filtering first was assumed to be always faster | Measured three filtering methods; picks one by how many passages survive | Benchmark below: identical results, crossover between 6% and 10% of passages |
| Evidence must be traceable | Keeps the raw source text beside the cleaned text; marks inferred labels | Every result shows its passage, label and caveats |
| Semantic search was assumed to beat keywords | Measured against a BM25 keyword baseline on 614 queries | Keyword search won on this test; see [Search quality](#search-quality) |

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

**Similar to a patent:** `--similar-to US20240051333A1` finds the patents closest to one already
in the dataset, using its abstract and first claim as the query. The patent itself is never
returned, every filter still applies, and no model is loaded, so it answers in a few
milliseconds. Patents with the very same abstract are flagged as possibly the same family (D20).

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

"Description para N" in results is the Nth non-blank paragraph of the description in this
dataset. The source has no official paragraph numbers, so N can differ from the `[00NN]`
numbering in the published document.

## Search quality

Search quality was tested twice, because the first test turned out to favour one method. The
first result was unexpected, so rather than explain it away I ran a second test to check why.

### Check 1: finding a patent from its own claim (automated, reproducible)

Each patent's first claim is used as the query, with that patent's claims hidden, so the
search can only find it through its abstract and description. 614 patents were tested (26 with
an identical twin were skipped); ranks are out of 640. Full results:
[`evaluation/self_retrieval.md`](evaluation/self_retrieval.md).

| Ranking | Found first (Recall@1) | In top 10 (Recall@10) | MRR |
|---|---:|---:|---:|
| Semantic (`all-MiniLM-L6-v2`, the current default) | 52.4% | 82.2% | 0.627 |
| BM25 keyword baseline | 86.5% | 93.8% | 0.893 |
| Combined (reciprocal rank fusion, untuned) | 70.0% | 89.9% | 0.769 |

Keyword matching clearly won. Part of that is the test itself: in 344 of the 614 patents, the
description repeats every distinctive word of the claim. But keyword search also won where some
of those words were missing (74.1% against 45.2% found first), so the general-purpose model,
trained on web text and reading only the first 256 tokens of a long claim, is genuinely weaker
at telling similar patents apart when the wording matches.

What this test can't show is the case semantic search exists for: a user describing an idea in
their own words, different from the patent's.

### Check 2: paraphrased queries (everyday language, exploratory)

To test that case, 12 technical information needs were written (for example, preventing wheel theft,
or tyres that inflate themselves), each in two wordings: everyday language, and wording close to
the patents' own terms. That gave 24 queries. The queries and grading rules were fixed before any
results were seen. Each method's results were pooled (355 patents in all) and graded 0–2 for
relevance, blind to which method returned each patent and at what rank. Scores are nDCG@10,
which rewards putting the most relevant patents near the top.

| Query wording | Semantic | BM25 | Combined |
|---|---:|---:|---:|
| Everyday language | **0.475** | 0.335 | 0.468 |
| Patent terminology | 0.836 | 0.807 | **0.844** |

- Semantic search beat BM25 on 8 of the 12 everyday-language needs. The biggest gaps came where
  the query and the patent shared almost no words: for wheel theft, the first relevant patent was
  at rank 2 with semantic search and rank 118 with BM25; for self-inflating tyres, rank 4 against 45.
- With patent terminology, all three methods were close.

**How far to trust it:** this is suggestive, not conclusive. It covers only 12 needs, and the 95%
interval for semantic search's advantage on everyday language is about −0.001 to +0.30, which
only just touches zero. Relevance was graded by an LLM (Claude), not by people, and the same
model helped write the queries; an LLM judges by meaning, which may lean toward semantic search.
Grading only covered pooled results, so relevant patents that no method found are missed. It was
run in a separate session, and its queries and grades are not in this repository, so unlike
check 1 it can't be reproduced from here.

### What the two checks show together

- **They agree rather than conflict.** BM25 is strong when the query uses the patent's own
  wording; semantic search is stronger when it doesn't.
- **Semantic search stays the default.** Natural-language descriptions of an invention are the
  main use case, and that is where check 2 favours it. Switching to BM25 on check 1 alone would
  have meant choosing the default on a test built in its favour.
- **Combining the two is never far from the best.** It nearly tied semantic search on everyday
  language, scored highest on patent terminology, and sat between the two on check 1. That makes
  it the natural next candidate for the default, but the gains are small, so it needs a larger,
  human-judged test before it's adopted (D18).

## Structured filtering and retrieval performance

The brief calls this *hybrid search*: semantic retrieval combined with strict constraints. The
benchmark runs 12 fixed queries, 30 times each, under 8 filter scenarios, and compares three
ways to apply the filters:

- **Prefilter:** find the eligible patents through the lookup tables, copy out only their
  vectors, and score those.
- **Mask:** find the eligible patents the same way, score every vector in place, and hide the
  rest.
- **Postfilter (naive):** rank every vector, then walk down the ranking checking each patent's
  text against the filters, with no lookup tables.

Median times per search once the query is embedded, on an 8-core Windows 11 laptop, plugged in
and idle, with the model already loaded (full tables in
[`benchmarks/results.md`](benchmarks/results.md)):

| Filter | Patents left | Share of pieces | Prefilter (ms) | Mask (ms) | Naive postfilter (ms) | Auto picks |
|---|---:|---:|---:|---:|---:|---|
| None | 640 | 100% | – | 2.48 | – | no filter |
| Classification B60B | 298 | 44.2% | 9.14 | **2.53** | 5.99 | mask |
| Title/abstract: tread | 156 | 28.9% | 6.22 | **2.56** | 7.33 | mask |
| Classification B60C11 | 101 | 19.1% | 4.32 | **2.42** | 5.88 | mask |
| B60C + title: pneumatic | 94 | 17.7% | 5.39 | **3.21** | 9.54 | mask |
| Abstract: tread groove | 45 | 9.5% | 2.83 | **2.59** | 13.88 | mask |
| Exact title "PNEUMATIC TIRE" | 33 | 6.4% | **2.88** | 3.07 | 10.32 | prefilter |
| Title/abstract: sensor | 32 | 5.2% | **2.10** | 2.71 | 21.22 | prefilter |

All three methods returned the same patents, in the same order, with the same evidence
passages, in every case.

**Latency as a user sees it.** Embedding the query takes a median of 8.1 ms, so a warm search
takes about 10 to 11 ms end to end. Starting the CLI adds a one-off 13 s to load the model and
0.3 s to load the index; a long-running service would pay that once.

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

### What changes at large scale

At 10 million patents the same design means about 540 million passages and 830 GB of vectors,
and exact search would take tens of seconds per query. Part 2 designs the system for that size:
OpenSearch for keyword search, compressed vector search and filters in one engine; PostgreSQL to
track every file and patent through ingestion; GPU workers for the one-off embedding. See
[Part 2](#part-2-10-million-patents).

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
python -m src.cli --similar-to US20240051333A1 --classification B60B
```

The fifth one matches nothing and shows the suggestions. The last one lists patents similar to
patent 20240051333 within classification B60B; the ID can be typed with or without the `US`
prefix, kind code, spaces or slashes. With no query, filtered patents are
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

### 6. Check search quality

```
python evaluation/self_retrieval.py
```

About a minute. It writes `evaluation/self_retrieval.md` and should reproduce the check 1 table
in Search quality exactly.

### 7. Run the tests

```
python -m pytest tests -q
```

56 tests. 46 check the cleaning rules (claim numbering, numeral separation, token limits), the
search (filters and their edge cases, filtering strategies, excerpts, suggestions, similar
patents), the BM25 baseline and the self-retrieval ranking, on tiny hand-made indexes; they need
neither the model nor the dataset. The other 10 test the Part 2 proof of concept and need
PostgreSQL running (`docker compose up -d`); without it they are skipped.

## Part 2: 10 million patents

**Design:** [`docs/PART2_DESIGN.md`](docs/PART2_DESIGN.md): the architecture, the search and
ingestion pipelines, failure handling, status tracking, a cost breakdown (about $7,400 a month in
an illustrative AWS scenario, with every input shown), the main challenges, and a build order.

**Proof of concept:** [`poc/`](poc/README.md): the ingestion pipeline, run on this project's 64
weekly files with PostgreSQL in Docker. It shows jobs claimed safely by several workers, a crashed
worker's job taken over once its lease expires, temporary errors retried, the same file or patent
never stored twice, bad records quarantined with their reason, and a `status` command.

```
docker compose up -d
python -m poc.db reset
python -m poc.register data/patent_data_small poc/fixtures
python -m poc.worker
python -m poc.status
```

Expected: 64 jobs `complete`, the deliberately corrupt file `failed`, the file with bad records
`complete_with_errors` with 3 records quarantined, and 641 unique patents. Step-by-step
demonstrations of each guarantee are in [`poc/README.md`](poc/README.md).

## Repository layout

```
src/load_data.py      load, clean and split patents into pieces
src/text_utils.py     word normalisation shared by indexing and filters
src/build_index.py    embeddings, embedding cache, filter lookup tables
src/search.py         search, filters, filtering strategies, excerpts, suggestions
src/cli.py            command-line interface
src/bm25.py           BM25 keyword ranking, the evaluation baseline
benchmarks/timing.py  filters-on vs filters-off benchmark
evaluation/self_retrieval.py   automated search-quality check
tests/                unit tests
DECISIONS.md          decision log with evidence
docs/PART2_DESIGN.md  Part 2: design for 10 million patents
poc/                  Part 2: ingestion proof of concept (register, worker, status)
docker-compose.yml    PostgreSQL for the proof of concept
```

## Limitations

- **Search quality evidence is limited.** The reproducible check is a proxy that favours keyword
  matching; the everyday-language check is small, LLM-graded and not in this repository. Neither
  is a human-judged prior-art evaluation. The model was chosen as a light CPU baseline, not as the
  best model for patents.
- Keyword filters match words, not synonyms; "search by meaning" covers that, across all text.
- Patents are ranked by their single best passage; a patent with many matching passages shows
  a count but doesn't rank higher (D16).
- Planned next: a choice of ranking (semantic, keyword or combined), backed by the checks
  above (D18). Deferred: fuzzy title
  matching, links that open the patent at the matching passage, and a "related by meaning"
  section after strict results (D15).

**What the data can support.** The sample is 640 vehicle-related US patent applications
published between February 2024 and May 2025. It cannot support real prior-art, novelty or
infringement conclusions, which need older patents, other countries, granted patents and
non-patent literature. The engine is not specific to vehicles: classification codes and
vocabulary come from the data.