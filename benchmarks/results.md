# Timing: filters on vs off

- Machine: Windows-11-10.0.26200-SP0, 8 logical CPUs, Python 3.12.10, NumPy 2.5.4
- Index: 640 patents, 34551 units, 384-dim vectors (53.1 MB), model `sentence-transformers/all-MiniLM-L6-v2`
- Start-up (one-off): index load 0.21 s, model load 9.53 s
- 12 queries x 30 repeats per cell (after 3 warm-up rounds), top 10 results
- Query embedding (same for every strategy, not included below): median 8.6 ms, p95 12.4 ms
- All strategies returned identical results: yes

## Search time after the query is embedded (ms)

| Scenario | Eligible patents | Eligible units | Share of units | prefilter median / p95 | mask median / p95 | postfilter (naive) median / p95 |
|---|---:|---:|---:|---:|---:|---:|
| no filters | 640 | 34551 | 100.0% | - | 2.26 / 2.73 (no filter) | - |
| classification B60B | 298 | 15263 | 44.2% | 8.92 / 10.62 | 2.22 / 2.74 | 5.44 / 7.18 |
| classification B60C11 | 101 | 6598 | 19.1% | 3.99 / 4.83 | 2.10 / 2.60 | 5.35 / 7.26 |
| title/abstract keyword: tread | 156 | 9994 | 28.9% | 5.77 / 7.41 | 2.22 / 2.76 | 6.62 / 17.53 |
| abstract keywords: tread groove | 45 | 3279 | 9.5% | 2.22 / 2.74 | 2.09 / 2.53 | 12.07 / 36.65 |
| title/abstract keyword: sensor | 32 | 1780 | 5.2% | 1.49 / 2.10 | 2.32 / 3.14 | 18.17 / 27.04 |
| B60C + title keyword: pneumatic | 94 | 6101 | 17.7% | 3.60 / 4.50 | 2.09 / 2.58 | 6.26 / 8.51 |
| exact title: "PNEUMATIC TIRE" | 33 | 2203 | 6.4% | 1.61 / 2.12 | 2.08 / 2.60 | 6.49 / 12.08 |

Unfiltered baseline: median 2.26 ms.

## Where the time goes (median ms per stage)

| Scenario | Strategy | Resolve filters | Score | Rank | Fetch evidence | Total |
|---|---|---:|---:|---:|---:|---:|
| no filters | no-filter | 0.004 | 2.03 | 0.17 | 0.045 | 2.26 |
| classification B60B | prefilter | 0.014 | 8.74 | 0.00 | 0.049 | 8.92 |
| classification B60B | mask | 0.017 | 1.88 | 0.16 | 0.040 | 2.22 |
| classification B60B | postfilter | 0.000 | 1.79 | 3.57 | 0.041 | 5.44 |
| classification B60C11 | prefilter | 0.011 | 3.86 | 0.00 | 0.047 | 3.99 |
| classification B60C11 | mask | 0.014 | 1.83 | 0.15 | 0.038 | 2.10 |
| classification B60C11 | postfilter | 0.000 | 1.81 | 3.41 | 0.041 | 5.35 |
| title/abstract keyword: tread | prefilter | 0.054 | 5.58 | 0.00 | 0.048 | 5.77 |
| title/abstract keyword: tread | mask | 0.081 | 1.87 | 0.15 | 0.038 | 2.22 |
| title/abstract keyword: tread | postfilter | 0.000 | 1.79 | 4.78 | 0.043 | 6.62 |
| abstract keywords: tread groove | prefilter | 0.099 | 2.02 | 0.00 | 0.045 | 2.22 |
| abstract keywords: tread groove | mask | 0.123 | 1.76 | 0.13 | 0.033 | 2.09 |
| abstract keywords: tread groove | postfilter | 0.000 | 1.69 | 10.28 | 0.040 | 12.07 |
| title/abstract keyword: sensor | prefilter | 0.048 | 1.34 | 0.00 | 0.042 | 1.49 |
| title/abstract keyword: sensor | mask | 0.066 | 2.00 | 0.14 | 0.037 | 2.32 |
| title/abstract keyword: sensor | postfilter | 0.000 | 1.83 | 16.24 | 0.044 | 18.17 |
| B60C + title keyword: pneumatic | prefilter | 0.085 | 3.41 | 0.00 | 0.043 | 3.60 |
| B60C + title keyword: pneumatic | mask | 0.106 | 1.75 | 0.14 | 0.035 | 2.09 |
| B60C + title keyword: pneumatic | postfilter | 0.000 | 1.71 | 4.45 | 0.037 | 6.26 |
| exact title: "PNEUMATIC TIRE" | prefilter | 0.013 | 1.51 | 0.00 | 0.040 | 1.61 |
| exact title: "PNEUMATIC TIRE" | mask | 0.017 | 1.83 | 0.14 | 0.035 | 2.08 |
| exact title: "PNEUMATIC TIRE" | postfilter | 0.000 | 1.78 | 4.64 | 0.041 | 6.49 |

Notes: for prefilter, copying out the eligible vectors, scoring and ranking are interleaved and reported together under Score. For postfilter, the filters are checked while walking down the ranking, so that cost is under Rank.
