# Part 2: 10 million patents

[← Back to the README](../README.md)

**Design:** [`docs/PART2_DESIGN.md`](PART2_DESIGN.md): the architecture, the search and
ingestion pipelines, failure handling, status tracking, a cost breakdown (about $7,400 a month in
an illustrative AWS scenario, with every input shown), the main challenges, and a build order.

**Decisions settled by measurement:** every passage is embedded, because 64% of top results
matched on a description paragraph (so skipping descriptions would lose evidence); OpenSearch is
used rather than PostgreSQL alone, because keyword ranking proved essential in Part 1. The cost
estimate is about $7,400 a month at 54 passages per patent, or about $16,500 at the 147 per
patent measured in Part 3, plus a one-off $31–188 to embed every passage on spot GPUs.

**Proof of concept:** [`poc/`](../poc/README.md): the ingestion pipeline, run on this project's 64
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
demonstrations of each guarantee are in [`poc/README.md`](../poc/README.md).

## What I saw when I ran it

I ran the proof of concept on a laptop with PostgreSQL in Docker, on the 64 weekly files plus the
two deliberately bad files in `poc/fixtures` (66 jobs). Every result below is from that run.

| Test | What happened |
|---|---|
| **One worker, all jobs** | 64 jobs `complete` (10 patents each), the corrupt file `failed` after one attempt with the JSON error, and the file with bad records `complete_with_errors`: 1 saved, 3 set aside, each with its reason. In total 644 records read, 641 saved, 641 unique patents, 0 conflicting versions |
| **Same files registered again** | Every file reported `already registered`; `0 new job(s) queued, 66 already registered` |
| **A worker crashes mid-job** | Worker A was stopped with Ctrl+C during job 1. Worker B finished the other jobs, and once A's 30-second lease ran out it took job 1: `attempt 2 (previous: lease of A expired)`. All 10 patents were saved |
| **Temporary errors** | With every job made to fail on its first attempt, each went to `retry_wait (next try in 5 s)` and completed on attempt 2. The corrupt file was not retried: a file that can never parse fails once instead of looping |
| **Two workers at once** | Between them they processed all 66 jobs, 35 in one window and 31 in the other, and no job appeared in both. The final counts were identical to the single-worker run: 644 read, 641 saved, 0 unchanged, 0 conflicts, 641 unique patents |

Two things worth noting. A crashed worker does not hold up the queue: the other worker keeps
going and takes the abandoned job as soon as its lease expires. And the last row is the real test
of "no duplicates": had both workers processed the same file, the second copy would have shown up
as `unchanged` or inflated the record count, and neither happened.

The 10 automated tests in `tests/test_poc.py` check the same behaviours, including cases that are
hard to trigger by hand, such as a lease expiring on the final attempt and a worker that lost its
lease trying to finish the job anyway.

