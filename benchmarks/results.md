# Timing: filters on vs off

- Machine: Windows-11-10.0.26200-SP0, 8 logical CPUs, Python 3.12.10, NumPy 2.5.4
- Index: 640 patents, 34551 units, 384-dim vectors (53.1 MB), model `sentence-transformers/all-MiniLM-L6-v2`
- Start-up (one-off): index load 0.26 s, model load 12.82 s
- 12 queries x 30 repeats per cell (after 3 warm-up rounds), top 10 results
- Query embedding (same for every strategy, not included below): median 8.1 ms, p95 10.8 ms
- All strategies returned identical results: yes

## Search time after the query is embedded (ms)

| Scenario | Eligible patents | Eligible units | Share of units | prefilter median / p95 | mask median / p95 | postfilter (naive) median / p95 |
|---|---:|---:|---:|---:|---:|---:|
| no filters | 640 | 34551 | 100.0% | - | 2.48 / 3.26 (no filter) | - |
| classification B60B | 298 | 15263 | 44.2% | 9.14 / 10.71 | 2.53 / 3.15 | 5.99 / 7.53 |
| classification B60C11 | 101 | 6598 | 19.1% | 4.32 / 5.18 | 2.42 / 2.93 | 5.88 / 8.06 |
| title/abstract keyword: tread | 156 | 9994 | 28.9% | 6.22 / 7.60 | 2.56 / 3.20 | 7.33 / 18.34 |
| abstract keywords: tread groove | 45 | 3279 | 9.5% | 2.83 / 3.53 | 2.59 / 3.21 | 13.88 / 43.99 |
| title/abstract keyword: sensor | 32 | 1780 | 5.2% | 2.10 / 2.78 | 2.71 / 3.46 | 21.22 / 30.66 |
| B60C + title keyword: pneumatic | 94 | 6101 | 17.7% | 5.39 / 7.40 | 3.21 / 4.54 | 9.54 / 15.25 |
| exact title: "PNEUMATIC TIRE" | 33 | 2203 | 6.4% | 2.88 / 4.01 | 3.07 / 4.17 | 10.32 / 19.29 |

Unfiltered baseline: median 2.48 ms.

## Where the time goes (median ms per stage)

| Scenario | Strategy | Resolve filters | Score | Rank | Fetch evidence | Total |
|---|---|---:|---:|---:|---:|---:|
| no filters | no-filter | 0.005 | 1.87 | 0.17 | 0.458 | 2.48 |
| classification B60B | prefilter | 0.018 | 8.55 | 0.00 | 0.443 | 9.14 |
| classification B60B | mask | 0.019 | 1.82 | 0.16 | 0.412 | 2.53 |
| classification B60B | postfilter | 0.000 | 1.80 | 3.71 | 0.435 | 5.99 |
| classification B60C11 | prefilter | 0.014 | 3.77 | 0.00 | 0.475 | 4.32 |
| classification B60C11 | mask | 0.014 | 1.75 | 0.15 | 0.433 | 2.42 |
| classification B60C11 | postfilter | 0.000 | 1.72 | 3.58 | 0.453 | 5.88 |
| title/abstract keyword: tread | prefilter | 0.068 | 5.58 | 0.00 | 0.489 | 6.22 |
| title/abstract keyword: tread | mask | 0.082 | 1.78 | 0.16 | 0.454 | 2.56 |
| title/abstract keyword: tread | postfilter | 0.000 | 1.74 | 5.01 | 0.483 | 7.33 |
| abstract keywords: tread groove | prefilter | 0.127 | 2.17 | 0.00 | 0.448 | 2.83 |
| abstract keywords: tread groove | mask | 0.140 | 1.77 | 0.16 | 0.441 | 2.59 |
| abstract keywords: tread groove | postfilter | 0.000 | 1.74 | 11.62 | 0.477 | 13.88 |
| title/abstract keyword: sensor | prefilter | 0.068 | 1.45 | 0.00 | 0.532 | 2.10 |
| title/abstract keyword: sensor | mask | 0.077 | 1.88 | 0.16 | 0.517 | 2.71 |
| title/abstract keyword: sensor | postfilter | 0.000 | 1.85 | 18.98 | 0.571 | 21.22 |
| B60C + title keyword: pneumatic | prefilter | 0.135 | 4.52 | 0.00 | 0.611 | 5.39 |
| B60C + title keyword: pneumatic | mask | 0.152 | 2.08 | 0.22 | 0.614 | 3.21 |
| B60C + title keyword: pneumatic | postfilter | 0.000 | 2.07 | 6.71 | 0.633 | 9.54 |
| exact title: "PNEUMATIC TIRE" | prefilter | 0.024 | 2.03 | 0.00 | 0.718 | 2.88 |
| exact title: "PNEUMATIC TIRE" | mask | 0.026 | 1.99 | 0.21 | 0.730 | 3.07 |
| exact title: "PNEUMATIC TIRE" | postfilter | 0.000 | 1.97 | 7.49 | 0.776 | 10.32 |

Notes: for prefilter, copying out the eligible vectors, scoring and ranking are interleaved and reported together under Score. For postfilter, the filters are checked while walking down the ranking, so that cost is under Rank.
