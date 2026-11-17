# Part 2: 10 million patents

[← Back to the README](../README.md)

**Design:** [`docs/PART2_DESIGN.md`](PART2_DESIGN.md): the architecture, the search and
ingestion pipelines, failure handling, status tracking, a cost breakdown (about $7,400 a month in
an illustrative AWS scenario, with every input shown), the main challenges, and a build order.

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
