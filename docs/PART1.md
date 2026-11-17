# Part 1: semantic patent search with strict filters

[← Back to the README](../README.md)

This page holds the full Part 1 write-up: what the engine does, how it works, how the data was
cleaned, how search quality was checked, how filtering was timed, and how to run everything.
Every design choice is written up in the [decision log](../DECISIONS.md), with its evidence and the
options rejected. This page states each reason in plain words; links labelled *reasoning* open
the full entry.

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
| Publication date | `--published-from 2024-06 --published-to 2025` | Published within the range, inclusive; a year, month or day each work |

Quoted words inside a filter (`'"wheel hub"'`) must appear together as a phrase. Filters
combine with AND, and they are strict: a patent that fails one never appears.

**Dates are publication dates.** The data has no filing or priority dates, only the weekly
publication date in each file's name ([reasoning](../DECISIONS.md#d3-the-weekly-date-is-the-publication-date-not-the-filing-date)), so the filter says "published". In prior-art work
the useful cut-off is "published before my priority date", which `--published-to` gives.
The large dataset's file names carry no dates, so its patents never pass a date filter ([reasoning](../DECISIONS.md#d26-a-publication-date-filter-and-newest-first-ordering)).

**Order:** results are ranked by score. `--sort newest` shows the same best matches newest
first; it reorders them, it does not fetch the newest patents overall. With no query, filtered
patents are always listed newest first.

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
milliseconds. Patents with the very same abstract are flagged as possibly the same family ([reasoning](../DECISIONS.md#d20-find-patents-similar-to-a-given-patent-id)).
Only patents in the dataset can be used; an unknown ID gets "not in this index" and the closest
IDs that are. Fetching outside patents (the brief's scraper idea) was left out on purpose: a
reliable local engine mattered more than a scraper that breaks when a website changes ([reasoning](../DECISIONS.md#d15-scope-what-was-deliberately-not-built)).

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
   vectors. The model is pinned to one exact version (`src/model.py`), so a later update
   under the same name can't quietly change the vectors or the results reported here. Lookup tables are built for the filters: for each word, which patents contain it in
   their title or abstract; for each classification prefix, which patents have it.
3. **Search** (`src/search.py`). The filters are resolved through the lookup tables, the query
   is embedded once and compared with the eligible pieces, each patent is ranked by its best
   piece, and the best pieces are shown as evidence.

Why pieces rather than whole patents: the model reads 256 tokens and a patent with a
description has a median of 4,195, so whole-patent vectors would judge each patent by its
first page. Pieces make every claim searchable and let the result show exactly what matched
([reasoning](../DECISIONS.md#d8-search-passages-rank-each-patent-by-its-best-passage)).

### Missing and malformed data

**Fields used:** `title`, `doc_number`, `abstract`, `claims`, `detailed_description` and
`classification`, plus `filename` for the Google Patents link; the publication date comes from
the weekly file's name (`patents_ipa240215.json`). The `bibtex` citation isn't used: it only
repeats the title, number and year found in other fields.

No field is missing outright, but 20,819 of 42,040 description paragraphs are empty strings
and 119 patents have no description text at all. **Every patent is kept;** blank paragraphs are
skipped, and patents without a description stay searchable through their title, abstract and
claims ([reasoning](../DECISIONS.md#d1-keep-every-patent-skip-blank-description-paragraphs)). Other data problems found and fixed (lost claim numbers, text too long for the model, and
others) are each written up with evidence in the [decision log](../DECISIONS.md#d2-take-claim-numbers-from-the-printed-number-never-from-list-position). The
weekly date in the file names is the A1 **publication** date, not the filing date ([reasoning](../DECISIONS.md#d3-the-weekly-date-is-the-publication-date-not-the-filing-date)).

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
[`evaluation/self_retrieval.md`](../evaluation/self_retrieval.md).

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
  human-judged test before it's adopted ([reasoning](../DECISIONS.md#d18-measure-search-quality-with-two-complementary-checks-keep-semantic-ranking-as-the-default)).

### What I would use to improve search quality

The checks point to a product decision that isn't built yet: **no single ranking is best for
every query.** A researcher who pastes a claim uses the patent's own wording, where keywords win;
one who describes an idea in everyday words is better served by meaning. [Part 2](PART2.md)'s design already
ranks with both combined. To improve search quality, I would let the user choose the ranking,
and pick the default with a test designed to settle it.

**What I would build.** A `--ranking semantic|keyword|combined` option in the CLI and a selector
on the page. All three modes would share the same passages, strict filters and best-passage rule,
so only the scoring differs; keyword ranking is already written (`src/bm25.py`), and combining is
the reciprocal rank fusion used in check 1. Each result would show the passages that scored
highest under the chosen mode. The combined score is a rank-based number (around 0.03), not a
similarity, so it would be labelled as such. Semantic would stay the default until the test below.

**Why it isn't built now.** Changing what every user sees on 12 AI-graded queries would be
deciding on thin evidence. Adding the option without the test would also push the choice onto
users without telling them when each mode helps.

**The test that would decide the default:**

- **More, realistic queries, split by type.** About 100 information needs in three kinds:
  everyday descriptions written by people who haven't read the patents, wording close to patent
  language, and pasted claims taken from patents outside the index (so the answer isn't the
  query's own source). The best ranking may differ by type, so results are reported per type,
  not only overall.
- **Human judges.** Two people with patent experience grade each pooled result 0–2, blind to
  method and rank, and their agreement is measured; an AI grader runs alongside only as a
  cross-check. This removes check 2's main weakness.
- **Pooling deep enough for recall.** The top 20 from each method are pooled for grading.
  Missing prior art is the costly failure ([Part 2](PART2.md)), so recall@20 is reported next to nDCG@10.
- **Tuning kept apart from testing.** The fusion's settings, and weights between the two
  methods, are chosen on a separate fifth of the queries, never on the test set.
- **A decision rule fixed in advance.** Combined becomes the default only if it is better than
  semantic overall, with a 95% bootstrap interval that excludes zero, and not clearly worse on
  any query type. If the types disagree (for example, keywords win on pasted claims), the
  default follows the query: pasted claims start in keyword or combined mode, descriptions in
  semantic mode.
- **The model in the same harness.** A patent-trained embedding model can be tested with the
  same queries and judges, since check 1 suggests part of the gap is the general-purpose model.

**Cost, roughly:** 100 queries with about 25 pooled patents each is about 2,500 judgements per
judge, around 20 hours at 30 seconds each. That is why it is a planned experiment rather than
part of this submission.

**Two more improvements, both about grouping results:**

- **Collapse patent families.** Results are individual patents, so versions of the same
  invention can take several places in a top 10: a similar-patent search returned a
  same-abstract twin at 0.985. Showing one entry per family ("+1 version with the same
  abstract") would give those places to different patents, which matters because missing
  prior art is the costly failure. The same-abstract flag ([reasoning](../DECISIONS.md#d20-find-patents-similar-to-a-given-patent-id)) is the starting point; only 26
  of the 640 patents (4%) have such a twin here, so the gain grows with a corpus that includes
  corrections and grants ([Part 2](PART2.md) keeps those as separate documents).
- **A landscape view.** A ranked list answers "what matches my idea?" but not "what is being
  filed in this area, and by whom?", which the brief lists as a reason to search patents.
  Grouping the relevant results by topic (clustering their vectors), classification and
  publication month would show that. "By whom" isn't possible with this data: the records have
  no applicant or assignee field, so it would have to be added at ingestion (Part 2's pipeline
  is where it would come in).

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
[`benchmarks/results.md`](../benchmarks/results.md)):

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
  on two machines (it moved between them, so it is a tuned value; [reasoning](../DECISIONS.md#d12-choose-the-filtering-method-by-how-many-units-survive)).
- **Exact search as one matrix multiplication** on normalised vectors: at about 2 ms, an
  approximate index would add complexity and lose recall for no visible gain ([reasoning](../DECISIONS.md#d9-all-minilm-l6-v2-with-exact-search-in-numpy)).
- **An embedding cache** keyed by model and exact text: after a cleaning change, a rebuild
  re-embeds only the changed pieces, about 5 minutes instead of 20 ([reasoning](../DECISIONS.md#d10-cache-embeddings-by-model-and-exact-text)).

### What changes at large scale

At 10 million patents the same design means about 540 million passages at this sample's 54 per
patent, or 1.47 billion at the 147 per patent measured on the large dataset ([Part 3](PART3.md)), and exact
search would take from tens of seconds to over a minute per query. [Part 2](PART2.md) designs the system for that size:
OpenSearch for keyword search, compressed vector search and filters in one engine; PostgreSQL to
track every file and patent through ingestion; GPU workers for the one-off embedding. See
[Part 2](PART2.md).

## How to run Part 1

Run every command from the repository's top folder.

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
python -m src.cli "tire pressure sensor" --published-from 2025 --sort newest
```

The fifth one matches nothing and shows the suggestions. The last one lists patents similar to
patent 20240051333 within classification B60B; the ID can be typed with or without the `US`
prefix, kind code, spaces or slashes. With no query, filtered patents are
listed newest first. `--excerpts N` changes how many passages each result shows, `--full`
prints them whole, and `python -m src.cli --help` lists every option.

**Or in a browser:** `streamlit run app.py` opens a simple page with the same search, filters,
similar-patent search and evidence passages. It is a convenience over the same engine, not the
[Part 3](PART3.md) interface option. The model loads once, so searches after the first take milliseconds
instead of paying the CLI's 13 s start-up each time.

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

72 tests. 62 check the cleaning rules (claim numbering, numeral separation, token limits), the
search (filters and their edge cases, the date filter and newest-first order, filtering
strategies, excerpts, suggestions, similar patents), the BM25 baseline, the self-retrieval ranking, large-data loading and HNSW search, on
tiny hand-made indexes; they need neither the model nor the dataset. The other 10 test the [Part 2](PART2.md) proof of concept and need
PostgreSQL running (`docker compose up -d`); without it they are skipped.
