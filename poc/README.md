# Part 2 proof of concept: reliable ingestion

A small, real version of the ingestion pipeline in
[`docs/PART2_DESIGN.md`](../docs/PART2_DESIGN.md) (sections 5 and 11). It shows the parts that are
hardest to get right: claiming jobs safely, retries, crash recovery, no duplicates, quarantine of
bad records, and status. It runs on Part 1's real weekly files.

It does **not** include OpenSearch, embeddings or anything about performance at 10 million patents.

| File | Role |
|---|---|
| `../docker-compose.yml` | PostgreSQL 16 on port 5433 |
| `schema.sql` | `ingestion_jobs`, `publication_versions`, `source_records` |
| `register.py` | A file arrives: one job per (file, checksum, pipeline version) |
| `worker.py` | Claims a job with `SKIP LOCKED` and a lease, parses with Part 1's loader, saves |
| `status.py` | What the pipeline contains and what state it's in |
| `fixtures/` | A file with three bad records, and a file that isn't valid JSON |

## Run it

From the project folder, with the virtual environment active:

```
pip install -r requirements.txt
docker compose up -d
python -m poc.db reset
python -m poc.register data/patent_data_small poc/fixtures
python -m poc.worker
python -m poc.status --jobs
```

`docker compose up -d` needs Docker Desktop running; wait a few seconds for PostgreSQL to start.
The connection string can be changed with `POC_DATABASE_URL`.

## What each guarantee looks like

| Guarantee | Try this | You should see |
|---|---|---|
| Same file twice → one job | Run the `register` command again | `0 new job(s) queued, 66 already registered.` |
| Two workers share the work | Start `python -m poc.worker --name A --slow 0.2` and `--name B --slow 0.2` in two terminals | Each job processed by exactly one of them |
| A crashed worker's job is retried | Start `python -m poc.worker --name A --slow 1`, press Ctrl+C mid-job, then start `--name B` | B waits for A's 30-second lease to expire, then finishes the job: `attempt 2 … (previous: lease of A expired)` |
| Temporary errors are retried | `python -m poc.worker --fail-on-attempt 1` on fresh jobs | `retry_wait (next try in 5 s)`, then `complete` on attempt 2 |
| A broken file fails, without pointless retries | Part of the normal run | `patents_ipa991230_corrupt.json` is `failed` after 1 attempt, with the JSON error |
| Bad records are kept out, the rest saved | Part of the normal run | `complete_with_errors (1 saved, 3 quarantined)`, each reason listed by `status` |
| The same patent again is not duplicated | Copy a weekly file under a new name, register it, run a worker | `complete (10 unchanged)`; the patent count doesn't change |

## How it works

- **Claiming:** a short transaction picks the next job with `FOR UPDATE SKIP LOCKED`, sets it to
  `processing`, and gives it a lease (`lease_until`, 30 s by default) and a new `lease_token`.
  The worker extends the lease every 5 seconds while it works.
- **Crash recovery:** a job whose lease has run out can be claimed again. Every write checks the
  `lease_token`, so a worker that lost its lease can't finish or fail the job afterwards.
- **No duplicates:** a file's patents, its per-record outcomes and the job's final state are written
  in **one transaction**, so a crash leaves either everything or nothing. Patents are keyed by
  (ID, content hash): the same content again is `unchanged`; the same ID with different content is
  kept as a `conflict`, never overwritten.

  In production the search index is a second system that can't share this transaction, which is
  why the design also relies on idempotent writes and reconciliation (design, sections 5.5 and 7).
- **Retries:** a temporary error moves the job to `retry_wait` with a growing delay (5 s, then 20 s);
  after 3 attempts it is `failed`. A file that isn't valid JSON fails at once.

## Tests

`tests/test_poc.py` has 10 tests, one per guarantee above plus the lease and retry edge cases. Each
runs in its own throwaway schema. They need PostgreSQL running and are skipped otherwise, so the
Part 1 tests still run without Docker.
