# Timing: filters on vs off

- Machine: Windows-11-10.0.26200-SP0, 8 logical CPUs, Python 3.12.10, NumPy 2.5.4
- Index: 640 patents, 34925 units, 384-dim vectors (53.6 MB), model `sentence-transformers/all-MiniLM-L6-v2`
- Start-up (one-off): index load 0.16 s, model load 13.59 s
- 12 queries x 30 repeats per cell (after 3 warm-up rounds), top 10 results
- Query embedding (same for every strategy, not included below): median 11.4 ms, p95 18.0 ms
- All strategies returned identical results: yes

## Search time after the query is embedded (ms)

| Scenario | Eligible patents | Eligible units | Share of units | prefilter median / p95 | mask median / p95 | postfilter (naive) median / p95 |
|---|---:|---:|---:|---:|---:|---:|
| no filters | 640 | 34925 | 100.0% | - | 2.84 / 3.33 (no filter) | - |
| classification B60B | 298 | 15573 | 44.6% | 11.14 / 14.41 | 2.56 / 3.53 | 7.07 / 9.74 |
| classification B60C11 | 101 | 6645 | 19.0% | 3.92 / 4.63 | 2.02 / 2.57 | 5.24 / 7.09 |
| title/abstract keyword: tread | 156 | 10003 | 28.6% | 6.16 / 10.01 | 2.45 / 3.30 | 6.81 / 18.21 |
| abstract keywords: tread groove | 45 | 3299 | 9.4% | 2.62 / 3.77 | 2.42 / 3.42 | 15.03 / 55.40 |
| title/abstract keyword: sensor | 32 | 1812 | 5.2% | 1.76 / 2.41 | 2.41 / 3.05 | 22.69 / 38.57 |
| B60C + title keyword: pneumatic | 94 | 6088 | 17.4% | 5.00 / 6.41 | 2.69 / 3.45 | 9.70 / 14.30 |
| exact title: "PNEUMATIC TIRE" | 33 | 2198 | 6.3% | 2.19 / 3.27 | 2.59 / 3.61 | 9.29 / 19.41 |

Unfiltered baseline: median 2.84 ms.

## Where the time goes (median ms per stage)

| Scenario | Strategy | Resolve filters | Score | Rank | Fetch evidence | Total |
|---|---|---:|---:|---:|---:|---:|
| no filters | no-filter | 0.005 | 2.44 | 0.27 | 0.071 | 2.84 |
| classification B60B | prefilter | 0.019 | 10.89 | 0.00 | 0.061 | 11.14 |
| classification B60B | mask | 0.022 | 2.09 | 0.22 | 0.054 | 2.56 |
| classification B60B | postfilter | 0.000 | 2.06 | 4.73 | 0.054 | 7.07 |
| classification B60C11 | prefilter | 0.011 | 3.80 | 0.00 | 0.043 | 3.92 |
| classification B60C11 | mask | 0.013 | 1.77 | 0.14 | 0.034 | 2.02 |
| classification B60C11 | postfilter | 0.000 | 1.72 | 3.41 | 0.036 | 5.24 |
| title/abstract keyword: tread | prefilter | 0.053 | 5.96 | 0.00 | 0.051 | 6.16 |
| title/abstract keyword: tread | mask | 0.086 | 2.08 | 0.16 | 0.040 | 2.45 |
| title/abstract keyword: tread | postfilter | 0.000 | 2.02 | 4.61 | 0.044 | 6.81 |
| abstract keywords: tread groove | prefilter | 0.123 | 2.36 | 0.00 | 0.052 | 2.62 |
| abstract keywords: tread groove | mask | 0.150 | 1.97 | 0.19 | 0.046 | 2.42 |
| abstract keywords: tread groove | postfilter | 0.000 | 1.94 | 12.99 | 0.048 | 15.03 |
| title/abstract keyword: sensor | prefilter | 0.062 | 1.58 | 0.00 | 0.052 | 1.76 |
| title/abstract keyword: sensor | mask | 0.084 | 2.01 | 0.19 | 0.046 | 2.41 |
| title/abstract keyword: sensor | postfilter | 0.000 | 1.98 | 20.68 | 0.054 | 22.69 |
| B60C + title keyword: pneumatic | prefilter | 0.134 | 4.70 | 0.00 | 0.062 | 5.00 |
| B60C + title keyword: pneumatic | mask | 0.163 | 2.12 | 0.23 | 0.057 | 2.69 |
| B60C + title keyword: pneumatic | postfilter | 0.000 | 2.12 | 7.43 | 0.061 | 9.70 |
| exact title: "PNEUMATIC TIRE" | prefilter | 0.017 | 2.04 | 0.00 | 0.055 | 2.19 |
| exact title: "PNEUMATIC TIRE" | mask | 0.025 | 2.22 | 0.22 | 0.055 | 2.59 |
| exact title: "PNEUMATIC TIRE" | postfilter | 0.000 | 2.20 | 6.72 | 0.057 | 9.29 |

Notes: for prefilter, copying out the eligible vectors, scoring and ranking are interleaved and reported together under Score. For postfilter, the filters are checked while walking down the ranking, so that cost is under Rank.
