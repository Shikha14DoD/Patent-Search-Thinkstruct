# Self-retrieval check: finding a patent from its own claim

Each of 614 patents was searched with its first claim as the query, with that patent's claims hidden, so it could only be found through its abstract and description. 26 patents were skipped because another patent has an identical abstract. Ranks are out of 640 patents; each patent is ranked by its best passage.

| Method | Recall@1 | Recall@10 | MRR | Median rank |
|---|---:|---:|---:|---:|
| Semantic (all-MiniLM-L6-v2) | 52.4% | 82.2% | 0.627 | 1 |
| BM25 keyword baseline | 86.5% | 93.8% | 0.893 | 1 |
| Combined (RRF of both) | 70.0% | 89.9% | 0.769 | 1 |

Per query, semantic search ranked the patent higher in 28 cases and BM25 in 257; the rest were ties.

## Split by word overlap

How much of the claim's distinctive wording (words rare in the corpus) also appears in the same patent's abstract and description. Recall@1 per group:

| Group | Queries | Semantic | BM25 | Combined |
|---|---:|---:|---:|---:|
| Description repeats every distinctive word | 344 | 58.1% | 96.2% | 79.7% |
| Some distinctive words missing | 270 | 45.2% | 74.1% | 57.8% |

The combined row fuses both rankings with reciprocal rank fusion (k = 60, the standard value, not tuned on this test).

This is a proxy: it shows whether the search connects a claim's legal wording to the plain description of the same invention, not whether results are relevant prior art. Descriptions often restate claims word for word, which favours BM25.

Query embedding: 614 claims in 31.0 s.
