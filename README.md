# Patent Evidence Explorer

Semantic patent search that shows the claim or paragraph behind every result, with strict
filters on classification, title and abstract.

**Current corpus:** 640 US patent applications from a vehicle-related sample (wheels, tires and
couplings), published February 2024 to May 2025. Nothing in the engine is specific to vehicles:
the classification codes and vocabulary come from the data.

Built for Thinkstruct's patent search engine assignment. This page summarises the three parts; each links
to its full write-up:

- [Part 1](docs/PART1.md): the search engine and its filters.
- [Part 2](docs/PART2.md): how the system would scale to 10 million patents, plus a working
  proof of concept of the ingestion pipeline.
- [Part 3](docs/PART3.md): speed measurements and improvements on the large dataset.

**How to run this project**

1. **Install.** You need Python 3.12 or newer. Create a virtual environment and install the
   packages with `pip install -r requirements.txt` (exact commands in [How to run](#how-to-run)).
2. **Add the data.** The patent data supplied with the assignment is not in this repository, so
   copy it in yourself: the 64 weekly JSON files go in `data/patent_data_small/` (Parts 1 and 2),
   and the large dataset goes in `data/patent_data_large/` (Part 3 only).
3. **Run the part you want to look at.** Each part has its own step-by-step instructions, with
   the output to expect:
   - **Part 1, search:** build the index (about 20 minutes), then search from the command line or
     the browser page. Steps: [How to run Part 1](docs/PART1.md#how-to-run-part-1).
   - **Part 2, proof of concept:** needs Docker for PostgreSQL. Steps:
     [Part 2 page](docs/PART2.md) (short version) and
     [poc/README.md, "Run it"](poc/README.md#run-it) (each guarantee shown one by one).
   - **Part 3, speed on the large dataset:** builds a larger index, then runs the benchmarks.
     Steps: [How to run Part 3](docs/PART3.md#how-to-run-part-3).

All three sets of commands are also collected in one place under [How to run](#how-to-run) below.

Every design choice, with the evidence behind it, is in [`DECISIONS.md`](DECISIONS.md), a log of
numbered decisions (D1, D2, …). Where this page cites one, it also states the reason in plain
words, so you never need to open the log to follow the page.

**Terms used on this page**

- **Passage:** one searchable piece of a patent: the abstract, a single claim, or a single
  description paragraph.
- **Embedding / semantic search:** a model turns each passage into a list of numbers so that
  passages with similar *meaning* end up close together, even when the words differ.
- **BM25:** the classic keyword-ranking formula used by traditional search engines. It rewards
  passages that contain the query's words.
- **Recall@1 / Recall@10:** how often the correct patent is the first result / is among the top
  10 results.
- **Exact search:** comparing the query with every passage. Slow on large data but always correct.
- **HNSW:** an index that works like a shortcut map, so a search visits only a small part of the
  data instead of all of it. Faster, but approximate.
- **Classification code:** the patent office's category code for a patent, such as B60B for vehicle wheels.
- **Weekly files:** the small dataset is 64 JSON files, one per week of publications.

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
with strict filters on classification code, title and abstract keywords, and exact title.**

It is a retrieval tool for a person to review, not a judge of novelty or infringement.

## The three parts at a glance

| Part | What was built | Headline result | Full write-up |
|---|---|---|---|
| 1. Search engine | Semantic search over every claim and paragraph, with strict filters and the matching passage shown as evidence | A filtered search costs about the same as an unfiltered one (2.1–3.2 ms against 2.48 ms); the obvious way of filtering (rank everything, then discard) was 2–9× slower | [docs/PART1.md](docs/PART1.md) |
| 2. At scale | A design for 10 million patents and a working ingestion proof of concept | Two workers never take the same job, a crashed worker's job is finished by another, failed jobs are retried and nothing is stored twice, all shown on the real weekly files | [docs/PART2.md](docs/PART2.md) |
| 3. Efficiency and large data | Processing and search measured on the large dataset, then improved | Searches without filters 2.7–3.4× faster at 75,018 passages (5.8× in the scaling test), returning the same results as exact search | [docs/PART3.md](docs/PART3.md) |

## Part 1: semantic search with strict filters

- **Search by meaning, with evidence.** 640 patents are split into 34,551 passages (the abstract,
  each claim, each description paragraph). A query can be a description, a pasted claim or a
  patent ID (similar patents), and every result shows the claim or paragraph that matched.
- **Strict filters combined with meaning-based search.** Filter by classification code prefix,
  title and abstract keywords, exact title and publication date. A patent that fails a filter
  never appears, and when nothing matches, the engine says which filter caused it and suggests
  fixes. Filters stay cheap because the index keeps a lookup table from each word and
  classification code to the patents that contain it, and a date range is found with two binary
  searches over patents sorted by date, instead of checking every patent on every search. A
  command-line tool (CLI) and a simple browser page use the same engine.
- **Messy data handled, not dropped.** Many claims had lost their printed number, so numbers are
  rebuilt (96.7% of "according to claim N" references now point to the right claim). Every passage
  is cut to fit the model's 256-token reading limit (a token is roughly a word or part of one),
  and every patent is kept.
- **Timed with and without filters.** Three ways of applying a filter were compared (filter
  first, score everything and hide the rest, or rank first and discard afterwards). The engine
  picks the fastest one depending on how many passages the filter leaves, so filtered searches
  cost about the same as unfiltered ones.
- **An unexpected finding, followed up.** When the query used the patent's own wording, keyword
  ranking (BM25) beat the semantic model: it put the right patent first 86.5% of the time against
  52.4%. When the query used everyday wording, semantic search did better. Semantic search
  stays the default; combining the two is the next candidate.

**[Read the full Part 1 write-up →](docs/PART1.md)** What the engine does and how it works, data
cleaning, the two search-quality checks and what I would add, the filtering benchmark and
efficiency commentary, and full run instructions.

## Part 2: implementation at scale

- **A design a new engineer could start building from:** Amazon S3 (file storage) holds the
  original files; PostgreSQL (a database) tracks each job and its status; worker programs process
  the jobs; OpenSearch (a search engine) provides keyword search, compressed vectors and filters
  in one place; and a FastAPI service (a web API) answers queries. The design also covers cost,
  failure handling, open challenges and a build order.
- **Decisions settled by measurement:** every passage is embedded, because 64% of top results
  matched on a description paragraph (so skipping descriptions would lose evidence); OpenSearch
  is used rather than PostgreSQL alone, because keyword ranking proved essential in [Part 1](docs/PART1.md).
- **Cost:** about $7,400 a month in an illustrative AWS scenario at 54 passages per patent, or
  about $16,500 at the 147 per patent measured in [Part 3](docs/PART3.md). A one-off embedding of every passage
  costs about $31–188 on "spot" GPUs (spare cloud GPUs rented at a discount).
- **Proof of concept:** a small working version of the ingestion pipeline: PostgreSQL in Docker
  with register, worker and status commands, run on the 64 weekly files. Two workers never take
  the same job, a crashed worker's job is finished by another, failed jobs are retried, nothing
  is stored twice, and bad records are set aside with their reason. 10 tests.

**[Read the full Part 2 summary →](docs/PART2.md)** How to run the proof of concept and what to
expect; the design itself is in [docs/PART2_DESIGN.md](docs/PART2_DESIGN.md).

## Part 3: efficiency on the large dataset

The enhancement I picked: measure how long it takes to process the large dataset (or a portion of
it) into the database and to search it, improve both, and record the improvements. Here "the
database" is [Part 1](docs/PART1.md)'s search index (the passage vectors and filter tables); "processing" means
splitting patents into passages and embedding them.

**Why I picked efficiency and large data.**
- **It is measured, not judged.** Every result is a time, or how closely the faster search agrees
  with exact search, so nothing depends on someone grading results by hand.
- **It is where a real system breaks first.** [Part 1](docs/PART1.md) has about a thousand patents; the [Part 2](docs/PART2.md)
  design targets 10 million. Before anything else, a deployment has to know how processing and
  search behave as the data grows.
- **It tests [Part 2](docs/PART2.md)'s design.** It replaces Part 2's estimates (such as passages per patent and
  embedding speed) with measurements on real data; the measured 147 passages per patent now feeds
  Part 2's cost section.

- **Processing, measured:** turning text into embeddings is 99.5% of the build time. The large
  data averages 147 passages per patent, so all ~471,000 patents would take about 26 days on a
  laptop CPU. Four ways of speeding this up were each tested on a complete build and none was
  faster, so the real lever is a GPU (a graphics chip suited to this work).
- **Search, improved:** searches without filters now use an HNSW index (a shortcut map, see the
  terms above) and then re-score the best patents exactly. Searches ran 2.7–3.4× faster at 75,018
  passages, growing to 5.8× in a test where the index size was increased step by step. Recall@10
  was 99–100% (almost always the same top 10 as exact search) and the evidence passages shown
  were identical. Searches with filters keep exact search.
- **Data problems caught:** a stray file that silently dropped patents, and 38% of the
  classification codes (in the sample checked) written as the text "N/A", which are now treated
  as missing.

**[Read the full Part 3 write-up →](docs/PART3.md)** What "the database" means here, the
processing measurements and the four attempts, the search results and scaling table, and how to
run every benchmark.

## Engineering highlights

| Problem found | What the engine does | Evidence |
|---|---|---|
| A third of claims lost their printed number, and cancelled claims shift the numbering | Rebuilds claim numbers from printed numbers and cancelled ranges; flags the ones it had to infer | 96.7% of "according to claim N" references point to the right claim; automated tests |
| The model reads only 256 tokens at a time; 17% of dependent claims were losing their own text | Splits text by the model's own token count, so a claim's own words always fit | 0 of 34,551 passages over the limit; automated tests |
| Filtering first was assumed to be always faster | Measured three filtering methods; picks one depending on how many passages survive the filter | Identical results from all three; filtering first only wins when the filter leaves fewer than roughly 6–10% of passages (see [Part 1](docs/PART1.md)) |
| Evidence must be traceable | Keeps the original source text beside the cleaned text; marks labels that were inferred | Every result shows its passage, its label and any caveats |
| Semantic search was assumed to beat keywords | Measured against a BM25 keyword baseline on 614 queries | Keyword search won on this test (a smaller everyday-language test leaned towards semantic search); see [Search quality](docs/PART1.md#search-quality) |
| Exact search gets slower as the index grows | HNSW index for searches without filters, then exact re-scoring of the top patents | 2.7–3.4× faster at 75,018 passages (5.8× in the scaling test), recall@10 99–100%, same evidence passages; see [Part 3](docs/PART3.md) |
| Processing was assumed to be speedable on a normal laptop CPU | Measured four alternatives on complete builds before changing anything | None made a real build faster, so the defaults stayed; see [Part 3](docs/PART3.md) |
| Results must be reproducible, and rebuilding the index takes about 20 minutes | Pins the exact model version, and caches each embedding under the model name plus the exact text, so a change to cleaning re-embeds only the passages whose text changed | A rebuild after a cleaning change took about 5 minutes instead of 20; see [Part 1](docs/PART1.md) |
| Several workers processing the same queue can collide or crash halfway | Each job is claimed with a time-limited lease, an abandoned job is picked up by another worker, and writes are safe to repeat, so a retry never creates a duplicate | Shown on the 64 weekly files, with 10 tests; see [Part 2](docs/PART2.md) |

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
python benchmarks/timing.py            # filtered vs unfiltered speed
python evaluation/self_retrieval.py    # search-quality check
python -m pytest tests -q              # automated tests

# Part 3: put the large dataset in data/patent_data_large/, then
python -m src.build_index --data data/patent_data_small data/patent_data_large:300 --out index_large
python benchmarks/search_large.py --index index_large
python benchmarks/search_scaling.py --index index_large
```

Building the [Part 1](docs/PART1.md) index takes about 20 minutes on a laptop CPU. macOS and Linux commands, the
[Part 2](docs/PART2.md) proof of concept and the other [Part 3](docs/PART3.md) benchmarks are in the full write-ups.

## Repository layout

```
src/load_data.py      load, clean and split patents into pieces
src/text_utils.py     word clean-up shared by indexing and filters
src/build_index.py    builds the index: embeddings, embedding cache, filter lookup tables
src/search.py         search, filters, filtering strategies, excerpts, suggestions
src/cli.py            command-line interface
app.py                simple browser page over the same engine (streamlit run app.py)
src/bm25.py           BM25 keyword ranking, used as the comparison baseline in the tests
benchmarks/timing.py  filters-on vs filters-off benchmark
benchmarks/*_large.py, build_ab.py, embed_speed.py, search_scaling.py
                      Part 3: processing and search benchmarks on the large dataset
benchmarks/large/     Part 3 results
src/fast_embed.py     Part 3: alternative (ONNX) encoder tried for speed, used only by embed_speed.py
evaluation/self_retrieval.py   automated search-quality check (find a patent from its own claim)
tests/                unit tests
README.md             this summary
docs/PART1.md, PART2.md, PART3.md   full write-up of each part
DECISIONS.md          decision log with evidence
docs/PART2_DESIGN.md  Part 2: design for 10 million patents
poc/                  Part 2: ingestion proof of concept (register, worker, status)
docker-compose.yml    PostgreSQL for the proof of concept
```

## Limitations

- **The evidence for search quality is limited.** The automated check (finding a patent from its
  own first claim) happens to favour keyword matching, so it is only a rough guide. The second
  check, using everyday-language queries, is small, was graded by an AI model (Claude) rather
  than by people, and is not included in this repository. Neither is a proper prior-art
  evaluation judged by humans. The embedding model (`all-MiniLM-L6-v2`) was chosen because it is
  small and runs on a laptop CPU, not because it is the best model for patents.
- **Keyword filters match words, not meanings.** A title filter for "tire" will not match a
  patent that says "tyre" or "pneumatic casing". Searching by meaning, which looks at all of a
  patent's text, covers that gap.
- **Patents are ranked by their single best passage.** A patent with many matching passages
  shows how many it has, but is not ranked higher for it. I kept it that way because changing
  the ranking formula needs a measurement of result relevance to justify it, and I did not have
  one for this choice ([full reasoning](DECISIONS.md#d16-show-up-to-three-matching-pieces-per-result-keep-ranking-by-the-best-piece)).
- **Searches with filters do not scale as well as searches without.** The faster HNSW index
  (explained in the terms above) was tested up to 75,018 passages. Searches with a filter still
  compare the query with every passage, so their time keeps growing with the index. At the
  10-million-patent scale of [Part 2](docs/PART2.md), the filtering would run inside OpenSearch
  instead.
- **Processing was not made faster on a CPU.** Four ideas were tried and none helped on a
  complete build; only a GPU changes this ([Part 3](docs/PART3.md)).
- **US and English only.** The corpus is US English publications and the embedding model is
  trained on English, so patents from other countries or in other languages would not be searched
  well, and there is no country or language filter yet. How I would widen this is in
  [Future scope](#future-scope-other-countries-and-languages).
- **Not built, on purpose.** Each was left out for a reason written in the decision log
  ([reasoning](DECISIONS.md#d15-scope-what-was-deliberately-not-built)): fuzzy title matching
  (finding titles despite typos); links that open the patent page at the matching passage (I
  tried Google Patents text links and they did not scroll to the text); and a "related by
  meaning" section shown after the strict results.
- **What I would add next:** a choice of ranking (meaning-based, keyword, or a combination),
  with the default picked by a test where people judge the results
  ([details](docs/PART1.md#what-i-would-use-to-improve-search-quality)).

**What the data can support.** The sample is 640 vehicle-related US patent applications
published between February 2024 and May 2025. It cannot support real conclusions about whether
an idea is new (prior art), or whether something infringes, because those need older patents,
patents from other countries, granted patents and non-patent literature such as journal papers.
The engine is not specific to vehicles: classification codes and vocabulary come from the data.

## Future scope: other countries and languages

The full plan is in [Part 2 design, section 12](docs/PART2_DESIGN.md#12-future-scope-other-countries-and-languages).
In short:

- **One ingestion job per source per publication day.** Patent offices publish on different days:
  the European Patent Office every Wednesday, while the weekly US files in this data are dated
  Thursdays. With several sources something new arrives most days, so the weekly build becomes a
  daily check that embeds only the new passages and adds them to the live index. [Part 2](docs/PART2.md)'s job
  queue already treats every batch as an independent job, so a new source is a new job type.
- **Record country and language on every patent**, and add filters for both (the fields are already
  in the [Part 2](docs/PART2.md) design).
- **Test a multilingual embedding model first**, on a small sample, with the same checks as [Part 1](docs/PART1.md),
  before re-embedding anything. Translating to English at ingestion and keeping one index per
  language are the alternatives, with their costs compared in the design. Keyword ranking also
  needs a per-language word splitter.
- **Group the same invention filed in several countries** (a patent family) into one result, and
  measure whether that helps or hides relevant patents.
- **Widen in small steps:** one more office and one more language first, then measure.

None of this is built or measured yet; it is a plan.
