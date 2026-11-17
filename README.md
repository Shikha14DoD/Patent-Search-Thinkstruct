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

Every design choice, with the evidence behind it, is in [`DECISIONS.md`](DECISIONS.md), a log of
numbered decisions (D1, D2, …). Where this page cites one, it also states the reason in plain
words, so you never need to open the log to follow the page.

## How to run this project

Do **Step 0** once, then run any of the three parts. Each part is a short numbered list in
[How to run](#how-to-run) further down this page:

1. [Step 0: set up](#step-0-set-up-once): install Python packages and add the data.
2. [Part 1: build the index and search](#part-1-build-the-index-and-search)
3. [Part 2: run the ingestion proof of concept](#part-2-run-the-ingestion-proof-of-concept) (needs Docker)
4. [Part 3: measure speed on the large dataset](#part-3-measure-speed-on-the-large-dataset)

Each part's own page has the same steps with more explanation and the output to expect:
[Part 1](docs/PART1.md#how-to-run-part-1), [Part 2](docs/PART2.md) and
[poc/README.md](poc/README.md#run-it), [Part 3](docs/PART3.md#how-to-run-part-3).

## Terms used on this page

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

| Part | What was built | Why | Key finding from running it | Full write-up |
|---|---|---|---|---|
| 1. Search engine | Semantic search over every claim and paragraph, with strict filters and the matching passage shown as evidence | A whole patent is too long to count as evidence, and a patent that fails a filter must never appear | A filtered search costs about the same as an unfiltered one (2.1–3.2 ms against 2.48 ms); the obvious way of filtering was 2–9× slower | [docs/PART1.md](docs/PART1.md) |
| 2. At scale | A design for 10 million patents, and a working ingestion proof of concept | Reliable ingestion is where a large system fails first, so that is the part I built and tested | Two workers never take the same job, a crashed worker's job is finished by another, and nothing is stored twice (641 unique patents) | [docs/PART2.md](docs/PART2.md) |
| 3. Enhancement picked: **efficiency and large data** | Processing and search measured on the large dataset, then improved | The result is measured, not judged; it is where a real system breaks first; and it tests Part 2's estimates with real numbers | Searches without filters 2.7–3.4× faster at 75,018 passages (5.8× in the scaling test), with the same results; no CPU speed-up for processing, so a GPU is the lever | [docs/PART3.md](docs/PART3.md) |

## Part 1: semantic search with strict filters

**How the code addresses the problem statement**

- **The same idea in different words:** `src/build_index.py` turns every passage into an embedding
  (a list of numbers, so similar *meaning* ends up close together), and `src/search.py` embeds the
  query and returns the closest patents. A query can be a description, a pasted claim or a patent
  ID.
- **A whole patent is not evidence:** 640 patents are split into 34,551 passages (the abstract,
  each claim, each description paragraph; `src/load_data.py`). Every result shows the passages
  that matched, labelled as claim or paragraph. Lost claim numbers are rebuilt (96.7% of
  "according to claim N" references resolve) and every passage fits the model's 256-token limit.
- **Strict filters:** classification code prefix, title and abstract keywords, exact title and
  publication date are applied *before* ranking in `src/search.py`, so a patent that fails a filter
  never appears. When nothing matches, the engine says which filter caused it and suggests fixes.
  A command-line tool (`src/cli.py`) and a browser page (`app.py`) use the same engine.

**What I found when running it**

- **The data is messier than it looks.** 119 of 640 patents have no description text, and 20,819 of
  42,040 description paragraphs are blank. Blank paragraphs are skipped, and those patents stay
  searchable through their title, abstract and claims.
- **The obvious way of filtering is slow.** Ranking everything and then discarding was 2–9× slower
  than no filter. I compared three methods and the engine picks the fastest one by how many
  passages the filter leaves (the switch is near 8%), so filtered searches cost 2.1–3.2 ms against
  2.48 ms unfiltered.
- **Keyword ranking beat meaning-based ranking on one test.** When the query reused the patent's own
  wording, keyword ranking (BM25) put the right patent first 86.5% of the time against 52.4%. On
  everyday wording, semantic search did better (8 of 12 test needs; a small, AI-graded test).
  Semantic search stays the default; combining the two is the next candidate.

**[Read the full Part 1 write-up →](docs/PART1.md)** Details of each finding:
[data problems](docs/PART1.md#missing-and-malformed-data),
[search quality](docs/PART1.md#search-quality) and
[filtering benchmark](docs/PART1.md#structured-filtering-and-retrieval-performance), plus how
the index is built and how to run it.

## Part 2: implementation at scale

**How the design and code address the problem statement**

- **The design** ([docs/PART2_DESIGN.md](docs/PART2_DESIGN.md)) takes Part 1's search to 10 million
  patents: Amazon S3 (file storage) holds the original files, PostgreSQL tracks each job and its
  status, workers process the jobs, OpenSearch provides keyword search, compressed vectors and
  filters in one place, and a FastAPI service answers queries. About $7,400 a month in an
  illustrative AWS scenario (about $16,500 at the 147 passages per patent measured in
  [Part 3](docs/PART3.md)), plus a one-off $31–188 to embed everything on spot GPUs.
- **The code** (`poc/`) builds the hardest part to get right, reliable ingestion, in miniature:
  `register.py` creates one job per file, `worker.py` claims jobs safely with a lease and retries
  failures, and `status.py` reports what the pipeline holds. It runs PostgreSQL in Docker on the
  64 weekly files and has 10 automated tests.

**What I found when running it** (66 jobs: the 64 weekly files plus 2 deliberately bad files)

- **Normal run:** 64 jobs complete; the corrupt file failed once, with its error, instead of being
  retried; the file with bad records saved 1 and set aside 3, each with its reason. In total 641
  unique patents.
- **A worker crashed mid-job:** I stopped worker A with Ctrl+C. Worker B finished the other jobs
  and took A's job once its 30-second lease ran out.
- **Temporary errors:** with every job made to fail once, each completed on its second attempt.
- **Two workers at once:** 35 and 31 jobs, no job processed twice, and the final counts were
  identical to the single-worker run, so nothing was stored twice.
- **Settled by measurement in the design:** every passage is embedded, because 64% of top results
  matched on a description paragraph; OpenSearch is used rather than PostgreSQL alone, because
  keyword ranking proved essential in [Part 1](docs/PART1.md).

**[Read the full Part 2 summary →](docs/PART2.md)** See
[what I saw when I ran it](docs/PART2.md#what-i-saw-when-i-ran-it) for the full results, and the
[design](docs/PART2_DESIGN.md) for the architecture, cost and build order.

## Part 3: efficiency on the large dataset

**The enhancement I picked: efficiency and large data.** Measure how long it takes to process the
large dataset (or a portion of it) into the database and to search it, improve both, and record
the improvements. ("The database" is [Part 1](docs/PART1.md)'s search index; "processing" means
splitting patents into passages and embedding them.)

**Why I picked it.**
- **It is measured, not judged.** Every result is a time, or how closely the faster search agrees
  with exact search, so nothing depends on someone grading results by hand.
- **It is where a real system breaks first.** [Part 1](docs/PART1.md) has about a thousand
  patents; the [Part 2](docs/PART2.md) design targets 10 million. A deployment has to know how
  processing and search behave as the data grows.
- **It tests [Part 2](docs/PART2.md)'s design.** It replaces estimates (such as passages per patent
  and embedding speed) with measurements on real data; the measured 147 passages per patent now
  feeds Part 2's cost section.

**How the code addresses the problem statement**

- **Measuring:** `benchmarks/profile_large.py`, `embed_speed.py`, `build_ab.py`,
  `search_large.py` and `search_scaling.py` time the build and the search on the large dataset.
  `src/load_data.py` samples it evenly (`--data data/patent_data_large:300`) and loads it defensively.
- **Improving search:** `src/build_index.py` builds an HNSW index (a shortcut map, see the terms
  above) once the index is large enough, and `src/search.py` uses it for searches without filters,
  then re-scores the best patents exactly.

**What I found when running it** (940 patents, 75,018 passages)

- **Processing is all embedding.** Turning text into embeddings is 99.5% of the build time. The
  large data averages 147 passages per patent, so all ~471,000 patents would take about 26 days
  on a laptop CPU.
- **Four speed-ups, none adopted.** Each was tested on a complete build, and none was faster. One
  looked 1.3–1.4× faster in a small test but 0.99× in a real build. The real lever is a GPU.
- **Search got faster.** Searches without filters now use an HNSW index (a shortcut map, see the
  terms above) and then re-score the best patents exactly: 2.7–3.4× faster, growing to 5.8× as
  the index grew, with recall@10 of 99–100% and the same evidence passages.
- **Filtered searches stay exact.** Forcing the faster index to work with filters dropped recall to
  as low as 33%, so the speed-up is not worth it there.
- **Data problems caught:** a stray file that silently dropped patents, and 38% of the
  classification codes sampled written as the text "N/A", now treated as missing.

**[Read the full Part 3 write-up →](docs/PART3.md)** Details:
[processing measured](docs/PART3.md#processing-measured),
[the four attempts](docs/PART3.md#processing-four-attempts-none-adopted) and
[search improved](docs/PART3.md#search-improved), plus how to run every benchmark.

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

Run every command from the **top folder of the repository** (the folder that contains
`README.md`). The commands below are for Windows PowerShell; the macOS and Linux differences are
noted where they matter.

### Step 0: Set up (once)

1. **Check your Python version.** You need Python 3.12 or newer:
   ```
   python --version
   ```
2. **Create a virtual environment and install the packages.** This takes a few minutes and about
   3 GB of disk space, because PyTorch is large:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
   On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.
   A Hugging Face warning about "unauthenticated requests" is harmless.
3. **Add the data.** The patent data supplied with the assignment is not stored in this
   repository. Create the folder `data/patent_data_small/` and put the **64 weekly JSON files**
   in it, directly inside the folder. Each file holds about 10 patents, which makes the **640
   patents** used in Parts 1 and 2 (64 files, 640 patents).

You only need to do Step 0 once. After that, pick a part below.

### Part 1: build the index and search

1. **Build the index.** This reads the 64 files, splits the patents into passages and embeds
   them. It takes about **20 minutes** on a laptop CPU:
   ```
   python -m src.build_index --data data/patent_data_small --out index
   ```
   It should finish with `index written to index/`. A line near the end should say
   `units longer than the model's 256-token limit: 0`.
2. **Run a search from the command line:**
   ```
   python -m src.cli "sensor that detects a flat tire while driving"
   ```
   You get the best-matching patents, each with the claim or paragraph that matched.
3. **Add filters** to narrow the results:
   ```
   python -m src.cli "wheel rim" --classification B60B --top 5
   python -m src.cli "tire pressure sensor" --published-from 2025 --sort newest
   ```
4. **Or use the browser page** (it does the same searches, with the filters in a side panel):
   ```
   streamlit run app.py
   ```
5. **Optional checks.** Each prints what it found; the tests need no data:
   ```
   python benchmarks/timing.py            # filtered vs unfiltered speed (a few minutes)
   python evaluation/self_retrieval.py    # search-quality check (about a minute)
   python -m pytest tests -q              # automated tests
   ```
   Without Docker running, 62 tests pass and 10 are skipped; those 10 belong to Part 2 and run
   once PostgreSQL is up (see Part 2).

All the search options, and the output to expect, are in
[docs/PART1.md](docs/PART1.md#how-to-run-part-1).

### Part 2: run the ingestion proof of concept

This part needs **Docker Desktop** installed and running. It does not need the index from Part 1,
only the 64 data files from Step 0.

1. **Start the database.** Part 2 keeps its job list in PostgreSQL, a database. Docker runs it for
   you in a ready-made container, so you do **not** install PostgreSQL yourself.
   - Install **Docker Desktop**, open it, and wait until it says it is running.
   - Then start the database:
     ```
     docker compose up -d
     ```
     This reads `docker-compose.yml`, downloads PostgreSQL 16.6 the first time (a minute or two),
     and starts it in the background. It uses user `poc`, password `poc` and database
     `patents_poc`, on port 5433 (not the usual 5432, so it cannot clash with a PostgreSQL you
     already have). The Part 2 scripts connect with these settings automatically.
   - Check that it is ready:
     ```
     docker compose ps
     ```
     The status should say `healthy`. If it says `starting`, wait a few seconds and run it again.
2. **Reset it and register the files.** Registering creates one job per file. The `poc/fixtures`
   folder adds two deliberately bad files to show how errors are handled:
   ```
   python -m poc.db reset
   python -m poc.register data/patent_data_small poc/fixtures
   ```
3. **Run a worker.** It claims the jobs one at a time and processes them, printing one line for
   each job:
   ```
   python -m poc.worker
   ```
4. **See the result:**
   ```
   python -m poc.status --jobs
   ```
   You should see 64 jobs `complete`, the corrupt file `failed`, the file with bad records
   `complete_with_errors` (3 records set aside), and 641 unique patents. (The 32,146 passages it
   reports are counted before long texts are split to fit the model's token limit, so the number
   is lower than Part 1's 34,551.)
5. **Try the failure cases (optional).** These are the guarantees Part 2 is about. For each one,
   first get fresh jobs with the two commands from step 2, then:
   - **Registering the same files twice makes no duplicate jobs.** Run the `register` command
     again (you do not need fresh jobs for this one). Each file is listed as
     `already registered`, and the last line is `0 new job(s) queued, 66 already registered.`
   - **Two workers share the work.** Open two PowerShell windows, activate the virtual
     environment in both, and start one worker in each, a second or two apart:
     `python -m poc.worker --name A --slow 0.2` and `python -m poc.worker --name B --slow 0.2`.
     Each job appears under only one of them (typically one window shows the odd-numbered jobs
     and the other the even-numbered ones). Afterwards, `python -m poc.status` still shows
     641 unique patents.
   - **A crashed worker's job is finished by another.** Run
     `python -m poc.worker --name A --slow 1`. As soon as it prints `job 1 attempt 1`, press
     **Ctrl+C**; it prints `stopped mid-job`. Then run `python -m poc.worker --name B`. B works
     through the other jobs, and about 30 seconds after A stopped (when A's lease runs out) it
     takes job 1 and prints `job 1 attempt 2 … (previous: lease of A expired)`.
   - **A temporary error is retried.** Run `python -m poc.worker --fail-on-attempt 1`. Every job
     prints `retry_wait (next try in 5 s)`, the worker prints `waiting: …`, and then each job
     completes on attempt 2. The corrupt file is not retried; it fails once, with the JSON error.
6. **When you are finished,** stop the database with `docker compose down`.

To see each guarantee demonstrated (two workers sharing jobs, a crashed worker being taken over,
retries, no duplicates), follow [poc/README.md](poc/README.md#what-each-guarantee-looks-like).
The design itself is in [docs/PART2_DESIGN.md](docs/PART2_DESIGN.md).

### Part 3: measure speed on the large dataset

Do **Step 0** first, and ideally build the Part 1 index too, because Part 3 reuses its embeddings.

1. **Add the large dataset.** Create `data/patent_data_large/` and unzip the large dataset into
   it. The batch files can stay in their subfolders; every `patents*.json` file is found.
2. **Build the larger index.** It uses the 640 Part 1 patents plus 300 patents sampled evenly
   from the large dataset (940 patents, 75,018 passages). About 40 minutes the first time, under
   a minute once embeddings are cached:
   ```
   python -m src.build_index --data data/patent_data_small data/patent_data_large:300 --out index_large
   ```
3. **Measure search speed.** These two give the main Part 3 result (searches without filters 2.7–3.4×
   faster, and how the speed-up grows with index size). About 5 minutes in total:
   ```
   python benchmarks/search_large.py --index index_large
   python benchmarks/search_scaling.py --index index_large
   ```
4. **Measure processing speed (optional).** These show where build time goes and the four speed-ups
   that were tried; together they take about 30 minutes:
   ```
   python benchmarks/profile_large.py --data data/patent_data_large
   python benchmarks/embed_speed.py --data data/patent_data_large
   python benchmarks/build_ab.py --data data/patent_data_large:20 --before 64:1 --after 64:4
   ```

Each benchmark writes its result to `benchmarks/large/`, replacing the committed file. Times
depend on your machine, but the speed-ups and the agreement with exact search should match.
The time of each command and what it measures is in
[docs/PART3.md](docs/PART3.md#how-to-run-part-3).

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

### What the data can support

The sample is 640 vehicle-related US patent applications
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
