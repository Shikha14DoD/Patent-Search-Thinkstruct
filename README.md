# Patent Evidence Explorer

Semantic patent search that shows the claim or paragraph behind every result, with strict
filters on classification, title and abstract.

**Current corpus:** 640 US patent application publications from a vehicle-domain sample
(wheels, tires and couplings), published February 2024 to May 2025. Nothing in the engine is
specific to vehicles: classification codes and vocabulary come from the data.

Built as a take-home project for Thinkstruct. Every design choice, with the evidence behind
it, is in [`DECISIONS.md`](DECISIONS.md). The design for 10 million patents and its proof of
concept are in [Part 2](docs/PART2.md); speed measurements and improvements on the
large dataset are in [Part 3](docs/PART3.md).
This page summarises the three parts; each links to its full write-up.

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

## The three parts at a glance

| Part | What was built | Headline result | Full write-up |
|---|---|---|---|
| 1. Search engine | Semantic search over every claim and paragraph, with strict filters and the matching passage shown as evidence | A filtered search costs about the same as an unfiltered one (2.1–3.2 ms against 2.48 ms); naive filtering was 2–9× slower | [docs/PART1.md](docs/PART1.md) |
| 2. At scale | A design for 10 million patents and a working ingestion proof of concept | Safe job claiming, crash recovery, retries and no duplicates, shown on the real weekly files | [docs/PART2.md](docs/PART2.md) |
| 3. Efficiency and large data | Processing and search measured on the large dataset, then improved | Unfiltered retrieval 2.7–3.4× faster at 75,018 passages (5.8× in the scaling test), with the same results as exact search | [docs/PART3.md](docs/PART3.md) |

## Part 1: semantic search with strict filters

- **Search by meaning, with evidence.** 640 patents are split into 34,551 passages (the abstract,
  each claim, each description paragraph). A query can be a description, a pasted claim or a
  patent ID (similar patents), and every result shows the claim or paragraph that matched.
- **Strict hybrid filters.** Classification prefix, title and abstract keywords, exact title and
  publication date. A patent that fails a filter never appears, and when nothing matches the
  engine says which filter failed and suggests fixes. A CLI and a simple browser page share the
  same engine.
- **Messy data handled, not dropped.** Claim numbers rebuilt after cancelled claims (96.7% of
  cross-references resolve), every passage fitted to the model's 256-token limit, and every patent
  kept.
- **Timed with and without filters.** Three filtering methods were compared; the engine picks the
  faster one by how many passages survive, so filters cost about the same as no filter.
- **An unexpected finding, followed up.** Keyword ranking (BM25) beat the semantic model when the
  query uses the patent's own wording (86.5% against 52.4% found first), while semantic search won
  on everyday wording. Semantic stays the default; combining the two is the next candidate.

**[Read the full Part 1 write-up →](docs/PART1.md)** What the engine does and how it works, data
cleaning, the two search-quality checks and what I would add, the filtering benchmark and
efficiency commentary, and full run instructions.

## Part 2: implementation at scale

- **A design an intern could start on:** S3 for the original files, PostgreSQL to track jobs and
  status, workers, OpenSearch for keyword search, compressed vectors and filters in one engine,
  and a FastAPI service, with a cost breakdown, failure handling, challenges and a build order.
- **Decisions settled by measurement:** every passage is embedded, because 64% of top results
  matched on a description paragraph; OpenSearch over PostgreSQL alone, because keyword ranking
  proved essential in Part 1.
- **Cost:** about $7,400 a month in an illustrative AWS scenario at 54 passages per patent, or about
  $16,500 at the 147 per patent measured in Part 3; a one-off embedding of every passage costs
  about $31–188 on spot GPUs.
- **Proof of concept:** PostgreSQL in Docker with register, worker and status commands, run on the
  64 weekly files: jobs claimed safely, a crashed worker's job finished by another, retries, no
  duplicates, and bad records quarantined with their reason. 10 tests.

**[Read the full Part 2 summary →](docs/PART2.md)** How to run the proof of concept and what to
expect; the design itself is in [docs/PART2_DESIGN.md](docs/PART2_DESIGN.md).

## Part 3: efficiency on the large dataset

The enhancement I picked: measure how long it takes to process the large dataset (or a portion of
it) into the database and to search it, improve both, and record the improvements.

**Why I picked efficiency and large data.**
- **It is measured, not judged.** Every result is a time or an agreement with exact search, so
  nothing depends on manual grading.
- **It is where a real system breaks first.** Part 1 has about 10^3 patents; about 10^7 are filed
  every year. Before anything else, a deployment has to know how processing and search behave as
  the data grows.
- **It tests Part 2's design.** It replaces Part 2's estimates (passages per patent, embedding
  rate) with measurements on real data; the measured 147 passages per patent now feeds Part 2's
  cost section.

- **Processing, measured:** embedding is 99.5% of build time; the large data averages 147 passages
  per patent, so all ~471,000 patents would take about 26 days on a laptop CPU. Four ways to
  speed it up were tested on complete builds and none was faster, so the lever is a GPU.
- **Search, improved:** an HNSW graph for searches without filters, with exact rescoring of the top
  patents: retrieval 2.7–3.4× faster at 75,018 passages, growing to 5.8× in the scaling test, with
  recall@10 of 99–100% and the same evidence passages. Filtered searches keep exact search.
- **Data problems caught:** a stray file that silently dropped patents, and 38% of classifications
  written as "N/A", now treated as missing.

**[Read the full Part 3 write-up →](docs/PART3.md)** What "the database" means here, the
processing measurements and the four attempts, the search results and scaling table, and how to
run every benchmark.

## Engineering highlights

| Problem found | What the engine does | Evidence |
|---|---|---|
| A third of claims lost their printed number, and cancelled claims shift the numbering | Rebuilds claim numbers from printed numbers and cancelled ranges; flags inferred ones | 96.7% of "according to claim N" references resolve; unit tests |
| The model reads 256 tokens; 17% of dependent claims were losing their own text | Splits text by the model's own token count; a claim's own words always fit | 0 of 34,551 units over the limit; unit tests |
| Filtering first was assumed to be always faster | Measured three filtering methods; picks one by how many passages survive | Benchmark below: identical results, crossover between 6% and 10% of passages |
| Evidence must be traceable | Keeps the raw source text beside the cleaned text; marks inferred labels | Every result shows its passage, label and caveats |
| Semantic search was assumed to beat keywords | Measured against a BM25 keyword baseline on 614 queries | Keyword search won on this test; see [Search quality](docs/PART1.md#search-quality) |
| Exact search slows down in step with the index | HNSW graph for searches without filters, with exact rescoring of the top patents | Retrieval 2.7–3.4× faster at 75,018 passages (5.8× in the scaling test), recall 99–100%, same evidence; see [Part 3](docs/PART3.md) |
| Processing was assumed to be speedable on a CPU | Measured four alternatives end to end before changing anything | None made a real build faster, so the defaults stayed; see [Part 3](docs/PART3.md) |

## How to run

Full instructions, with expected output and timings: [Part 1](docs/PART1.md#how-to-run-part-1)
and [Part 3](docs/PART3.md#how-to-run-part-3). The short version, on Windows PowerShell (Python
3.12 or newer):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Part 1: put the 64 weekly JSON files in data/patent_data_small/, then
python -m src.build_index --data data/patent_data_small --out index
python -m src.cli "sensor that detects a flat tire while driving"
python -m src.cli "wheel rim" --classification B60B --top 5
streamlit run app.py
python benchmarks/timing.py
python evaluation/self_retrieval.py
python -m pytest tests -q

# Part 3: put the large dataset in data/patent_data_large/, then
python -m src.build_index --data data/patent_data_small data/patent_data_large:300 --out index_large
python benchmarks/search_large.py --index index_large
python benchmarks/search_scaling.py --index index_large
```

The Part 1 index takes about 20 minutes to build on a laptop CPU. macOS and Linux commands, the
Part 2 proof of concept and the other Part 3 benchmarks are in the full write-ups.

## Repository layout

```
src/load_data.py      load, clean and split patents into pieces
src/text_utils.py     word normalisation shared by indexing and filters
src/build_index.py    embeddings, embedding cache, filter lookup tables
src/search.py         search, filters, filtering strategies, excerpts, suggestions
src/cli.py            command-line interface
app.py                simple browser page over the same engine (streamlit run app.py)
src/bm25.py           BM25 keyword ranking, the evaluation baseline
benchmarks/timing.py  filters-on vs filters-off benchmark
benchmarks/*_large.py, build_ab.py, embed_speed.py, search_scaling.py
                      Part 3: processing and search benchmarks on the large dataset
benchmarks/large/     Part 3 results
src/fast_embed.py     Part 3: ONNX Runtime encoder, used only by embed_speed.py
evaluation/self_retrieval.py   automated search-quality check
tests/                unit tests
README.md             this summary
docs/PART1.md, PART2.md, PART3.md   full write-up of each part
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
- HNSW was measured up to 75,018 passages. Filtered searches still use exact search, so their
  time keeps growing with the index; at Part 2's scale the filters run inside OpenSearch.
- Processing was not made faster on a CPU; only a GPU changes that (Part 3).
- What I would add: a choice of ranking (semantic, keyword or combined), with the default
  chosen by a human-judged test ([details](docs/PART1.md#what-i-would-use-to-improve-search-quality)).
  Deferred: fuzzy title
  matching, links that open the patent at the matching passage, and a "related by meaning"
  section after strict results (D15).

**What the data can support.** The sample is 640 vehicle-related US patent applications
published between February 2024 and May 2025. It cannot support real prior-art, novelty or
infringement conclusions, which need older patents, other countries, granted patents and
non-patent literature. The engine is not specific to vehicles: classification codes and
vocabulary come from the data.
