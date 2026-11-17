# Engineering decision log

Each significant choice is recorded with the evidence behind it, the alternatives considered,
and what it costs. Decisions that changed after a measurement keep their history: the
reversal is part of the record.

**Status values:** *Validated* (backed by a measurement on this dataset), *Provisional*
(reasonable, but not yet measured), *Deferred* (deliberately not built yet), *Rejected*.

**Where the numbers come from.** Dataset figures: the 640-patent sample
(`data/patent_data_small`, 64 weekly files, February 2024 to May 2025), measured by the loader
(`python src/load_data.py`) and the index build (`index/build_stats.json`). Timings:
`benchmarks/results.md`, measured on an 8-core Windows 11 laptop with Python 3.12 and NumPy 2.5.
Token counts use the model's own tokenizer.

Decisions D1 to D7 concern the data; D8 to D14 and D16 onwards concern search; D15 records scope; D18 measures search quality; D21 to D25 are Part 3 (efficiency on the large dataset, results in `benchmarks/large/`).

---

## Data

### D1. Keep every patent; skip blank description paragraphs
- **Status:** Validated
- **Problem:** The brief says some patents have missing fields and asks us to choose and document how to handle them.
- **Evidence:** No field is missing outright, but 20,819 of 42,040 description paragraphs are empty strings, and 119 patents have no description text at all.
- **Alternatives:** Drop patents with no description; index blank paragraphs anyway.
- **Chosen:** Keep all 640 patents. Skip blank paragraphs. Patents without a description remain searchable through title, abstract and claims.
- **Rationale:** Dropping 119 patents (19% of the sample) would hide patents that may well be relevant; their claims are often the most important text anyway. Blank paragraphs carry nothing to search.
- **Trade-offs:** Those 119 patents can never show a description paragraph as evidence.
- **Validation:** The ingest report counts blank paragraphs and patents without a description on every build.

### D2. Take claim numbers from the printed number, never from list position
- **Status:** Validated (revised: the first version used list position)
- **Problem:** Every result shows which claim matched. A wrong claim number is worse than none, especially for a legal user.
- **Evidence:** 3,294 of 10,578 claim entries carry no printed number, and 127 entries are cancelled ranges such as `1 - 5 . (canceled)`. List position and the real number disagree for 1,131 claims. Example: in patent 20240051338, claims 1 to 5 are cancelled, so the first real claim is claim 6; the first version labelled it "Claim 1", and every label in that patent was off by five.
- **Alternatives:** List position (the first version); drop unnumbered claims.
- **Chosen:** Walk each patent's claims in order. Use the printed number when present; let cancelled ranges advance the count and skip them as content; otherwise infer *previous + 1* and flag the claim as "claim number inferred".
- **Rationale:** The sequence is recoverable from the surrounding numbered claims, and flagging keeps the result honest about what was reconstructed.
- **Trade-offs:** 3,223 claims carry an inferred number. An inference error would mislabel a claim.
- **Validation:** Dependent claims cite their parent by number ("according to claim 6"), which gives an independent check. After parsing, 96.7% of those citations point at a claim that exists, and 4,300 point at numbers that were inferred. The patents' own cross-references agree with the reconstruction. Unit tests cover the cancelled-range case.
- **Limitations:** 3.3% of citations still point at a claim number that doesn't exist in the parsed patent (237 references), likely conversion damage in the source.

### D3. The weekly date is the publication date, not the filing date
- **Status:** Validated (revised: the first version called it `filing_date`)
- **Problem:** Date filters and displays must say what the date actually is.
- **Evidence:** File names such as `US20240051333A1-20240215.XML`: A1 marks a pre-grant publication, and the date is when it was published. Applications usually publish about 18 months after filing.
- **Chosen:** Store and display it as `publication_date`.
- **Trade-offs:** No filing date is available in this data.

### D4. Separate reference numerals glued to words, in descriptions only
- **Status:** Validated
- **Problem:** Text like `The spoke100includes an axle body110` is unreadable as evidence and tokenizes into nonsense words.
- **Evidence:** 95,905 occurrences, all in the detailed description. Titles, abstracts and claims contain none.
- **Alternatives:** Leave it; separate every letter-digit boundary everywhere.
- **Chosen:** Insert spaces around numerals and after `FIG.`/`FIGS.` in description text, leaving measurement units (`205mm`, `32psi`) and codes (`B60B27/00`, `3D`) intact through an exception list.
- **Rationale:** Better input for the embedding model and readable excerpts. Restricting it to descriptions means the title and abstract keyword filters are unaffected.
- **Trade-offs:** An exception list can miss rare units.
- **Validation:** Unit tests for separation and for units and codes left intact.
- **Limitations:** A numeral glued after a closing bracket (`wheel)1010`) is not separated yet.

### D5. Keep raw text alongside cleaned text
- **Status:** Validated
- **Problem:** Cleaning is lossy; a reviewer must be able to see what the source said.
- **Chosen:** Each unit stores the embedded text, the displayed text and the raw source text. Notes mark reconstructed or added content: "claim number inferred", "opening text missing in source", and "with Claim N as context".
- **Limitation:** "Description para N" is the Nth non-blank paragraph of the description in this dataset. The source carries no official paragraph numbers (none of the 21,221 non-blank paragraphs starts with a `[00NN]` marker), so N can differ from the published document's numbering; 24 short heading-like lines are counted too. The README says so.

### D6. Don't index description paragraphs under five words
- **Status:** Validated
- **Problem:** A two-word fragment matches a short query almost perfectly and outranks real evidence.
- **Evidence:** For the query "wheel rim" with classification B60B, the top result's evidence was the fragment "vehicle wheel.", ahead of the patent titled "RIM FOR A WHEEL". 170 paragraphs are under five words (formula lines such as `B/A=1.00.`, stray fragments).
- **Alternatives:** Remove repeated boilerplate (rejected: repeated text can still be technically relevant); merge fragments into neighbouring paragraphs.
- **Chosen:** Skip paragraphs under five words. Paragraph numbering stays stable, so "Description para 12" still means the 12th non-blank paragraph.
- **Validation:** After the change, "RIM FOR A WHEEL" ranks first for that query.
- **Trade-offs:** A genuinely meaningful four-word paragraph would be lost; none was observed among the 170.

### D7. Every embedded text fits the model's 256-token input limit
- **Status:** Validated (revised: the first version split text by word count)
- **Problem:** all-MiniLM-L6-v2 reads 256 tokens and silently drops the rest.
- **Evidence:** Measured with the model's tokenizer on the first index: 1,406 of 10,451 claims (13.5%), 414 of 23,834 description chunks (1.7%) and 17 of 640 abstracts (2.7%) were cut off. Worse, 1,165 of 6,864 dependent claims (17%) lost part of their *own* text, because the parent claim was placed in front of it and the cut fell on the dependent claim's distinctive wording.
- **Alternatives:** Word-count splitting (the first version, which underestimated patent text's token density); dropping parent context entirely; a model with a longer input limit.
- **Chosen:** Split by token count, measured with the model's tokenizer. A claim's own text always fits; the parent claim (or title) is trimmed to the space left, and dropped if under 16 tokens remain. Long claims and abstracts are split into labelled parts like paragraphs.
- **Validation:** The build now counts units over the limit: 0 of 34,551. 2,150 texts were split, 1,207 parent contexts trimmed, 6 dropped. Unit tests check that every unit fits and that a dependent claim's own text survives. For five test queries, 8 to 10 of the top 10 results were unchanged, consistent with the fix mainly affecting claims.

---

## Search

### D8. Search passages, rank each patent by its best passage
- **Status:** Validated for evidence; relevance measured in part (D18)
- **Problem:** Choose the unit that is embedded and compared with the query.
- **Evidence:** A patent with a description has a median of 4,195 description tokens (90th percentile 10,303). Only 1.7% of whole patents would fit in 512 tokens, none in 256. By contrast, a single claim has a median of 42 tokens, a description chunk 103 and an abstract 147.
- **Alternatives:** (A) whole patents: the model would see only the first 256 tokens, about 6% of a typical patent. (B) Patent-level retrieval, then evidence extraction: smaller index, but the first stage can miss a patent whose only relevant passage is a single claim. (C) Separate abstract and passage indexes: duplication and more complex ranking, without a measured need.
- **Chosen:** Index abstracts, claims and description chunks (34,551 units, 54 per patent). A patent's score is its best unit's score, and that unit is the evidence shown.
- **Rationale:** It is the only option where the evidence shown is the text that actually matched, and no part of a patent is invisible to the search.
- **Trade-offs:** 54 vectors per patent instead of one. At 10 million patents that is roughly 540 million vectors, about 830 GB as 384-dimensional float32, so Part 2 needs approximate search and compression. Long patents get more chances at a high-scoring passage.
- **Validation:** Relevance measured in part by D18's two checks; a human-judged evaluation is still needed (D15).

### D9. all-MiniLM-L6-v2 with exact search in NumPy
- **Status:** Provisional (model); Validated (exact search at this scale)
- **Problem:** Choose an embedding model and a vector search method.
- **Evidence:** A full unfiltered search takes a median of 2.48 ms once the query is embedded, of which scoring all 34,551 vectors is 1.87 ms; embedding the query takes a median of 8.1 ms. The vectors take 53 MB. Building the embeddings ran at 20 to 29 texts per second on the laptop's CPU, about 20 minutes for the full sample.
- **Alternatives:** BGE-small (same size, often stronger on retrieval benchmarks); patent-specific models; FAISS or another approximate index.
- **Chosen:** all-MiniLM-L6-v2: small, fast on CPU, widely used, and quick for a reviewer to reproduce. Exact search with one matrix multiplication.
- **Rationale:** At about 2.5 ms, an approximate index would add infrastructure and a recall loss to save almost nothing, since query embedding dominates the request.
- **Trade-offs:** The model isn't tuned for patent language. Exact search grows linearly; around a million vectors it would reach tens of milliseconds.
- **Update (Part 3):** at 75,018 passages, unfiltered searches use an HNSW graph (D24); filtered searches keep exact search (D25).
- **Not done, and why:** Comparing models requires a relevance metric. D18 now provides one, and shows the model is weaker than BM25 when wording matches, but a model swap is still deferred: the reproducible check favours keywords, so it can't rank models fairly on its own.

### D10. Cache embeddings by model and exact text
- **Status:** Validated
- **Problem:** Embedding the sample takes about 20 minutes; every change to a cleaning rule would force a full rebuild.
- **Chosen:** Store each vector under a hash of (model name, exact embedded text), in an append-only file written in checkpoints.
- **Evidence:** After the D6 change, a rebuild embedded nothing new. After D7, it re-embedded 6,092 changed texts and reused 25,821, about 5 minutes instead of 20. An interrupted build resumes from the last checkpoint.
- **Alternatives:** No cache; a key that includes a preprocessing version number (rejected: the key is already the exact embedded text, so any cleaning change produces a new key on its own).
- **Limitations:** Stale vectors stay in the cache file; it only grows. A change to how the model is run (for example normalisation) would need the cache cleared.

### D11. Filters are strict, exact, and resolved through indexes
- **Status:** Validated
- **Problem:** The brief requires classification constraints, keywords in title or abstract, and specific-title search. A filter must never show a patent that doesn't satisfy it.
- **Chosen:**
  - Classification: prefix match (`B60B`, `B60C11`); several prefixes are OR-ed. Every prefix of every code is indexed, so a lookup is one dictionary hit. Codes come from the data; nothing is specific to vehicles.
  - Keywords: separate inverted indexes for title, abstract, and title-or-abstract. Matching is word-level with stemming (sensor/sensors, braking/brake) and tyre/tire variants; quotes match an exact phrase; all words by default, "any word" optional.
  - Exact title: case and punctuation normalised.
  - All filters are AND-ed. An empty result names the filter that matched nothing.
- **Evidence:** Resolving filters takes 0.005 to 0.15 ms in every benchmark scenario.
- **Alternatives:** A semantic fallback for filters (return "related" patents when no exact match exists). Rejected: a filter is a yes/no decision, and similarity has no natural cut-off, so any threshold would be arbitrary and the results hard to explain. Meaning-based search belongs in the ranked query.
- **Limitations:** Keyword filters don't match synonyms ("car" vs "vehicle"). Classification codes are stored without separators (`B60B2700FI`), so prefixes deeper than the subclass can be ambiguous. Generic titles are common: 33 patents are titled exactly "PNEUMATIC TIRE".

### D12. Choose the filtering method by how many units survive
- **Status:** Validated (revised twice)
- **Problem:** The brief says naive hybrid search is expensive and asks for timings with and without it.
- **History:** The plan assumed that filtering first and scoring only the survivors (prefilter) would always be faster. The first single runs showed a filtered search slower than an unfiltered one, so the benchmark compared three methods: prefilter (copy out and score only surviving units), mask (score all units in place, discard failures) and postfilter (the naive version: rank everything, then check filters by reading text). A provisional threshold of 25% came from a 2-core sandbox; the laptop measurements moved it to 8%.
- **Evidence (laptop, median ms after the query is embedded):**

  | Surviving units | Prefilter | Mask | Postfilter (naive) |
  |---|---:|---:|---:|
  | 100% (no filter) | – | 2.48 | – |
  | 44.2% (B60B) | 9.14 | 2.53 | 5.99 |
  | 19.1% (B60C11) | 4.32 | 2.42 | 5.88 |
  | 9.5% (abstract: tread groove) | 2.83 | 2.59 | 13.88 |
  | 6.4% (exact title) | 2.88 | 3.07 | 10.32 |
  | 5.2% (title/abstract: sensor) | 2.10 | 2.71 | 21.22 |

  The naive method is 2 to 9 times slower than no filter. Mask costs the same as no filter in every case. Prefilter wins only when a small share survives: copying out thousands of scattered vectors costs more than one multi-threaded multiplication over the whole contiguous matrix. All three return identical results in every case.
- **Chosen:** `auto`: prefilter when at most 8% of units survive, mask otherwise. 8% lies between the laptop's crossover (between 6.4% and 9.5%) and stays safe on the 2-core machine, where prefilter kept winning up to about 20%.
- **Trade-offs:** The crossover depends on hardware, so 8% is a tuned value, not a property of the method.
- **Measurement note:** One run with the laptop busy doubled every stage at once, embedding and scoring included (unfiltered median 5.41 ms, p95 14.30 ms). It was discarded and re-run with the laptop plugged in and idle; the conclusions were the same in both runs.
- **Implication for scale:** At this size, filtering can't make search much faster; it can only avoid making it slower. Its real value appears at large scale, where an index partitioned by classification would let a filter skip whole partitions without copying (Part 2).

### D13. Break score ties deterministically
- **Status:** Validated
- **Problem:** The same query must give the same results whichever filtering method runs it.
- **Evidence:** Identical passages appear in several patents (11 groups of identical abstracts covering 26 patents; 416 paragraphs appear in three or more patents), so those patents get exactly the same score. The benchmark's consistency check found 6 of 84 query-filter combinations where methods disagreed on the 10th result.
- **Chosen:** Keep every patent tied with the k-th score, then break ties by index position.
- **Validation:** All 84 filtered query-scenario combinations (12 queries x 7 filter scenarios) now agree across methods.
- **Related:** Patents with identical abstracts are not merged; identical text doesn't prove they are the same patent family.

### D14. Semantic ranking only for now; keyword (BM25) fusion deferred
- **Status:** Deferred; BM25 baseline and fusion since measured (D18)
- **Problem:** Embeddings match meaning; keyword ranking matches exact terms. Combining them may help.
- **Evidence:** "wheel rim" is the kind of query where keyword ranking helps: the query's exact words are a patent's title. Semantic search still ranks that patent first after D6, but by a small margin (0.703 vs 0.698).
- **Why deferred:** Without a relevance metric there is no way to show whether fusion helps or hurts, and fusion adds a tuning parameter. A measured BM25 baseline belongs with the evaluation work.

---

## D15. Scope: what was deliberately not built

| Item | Decision | Reason |
|---|---|---|
| Fuzzy title matching | Deferred | Exact normalised matching meets the brief; fuzzy matching needs tuning and a test set. |
| Million-vector synthetic benchmark | Deferred | Part 2 or Part 3; the real crossover data above already answers the brief's question at this size. |
| Relevance evaluation | Partly done (D18) | An automated self-retrieval check with a BM25 baseline is in the repo, and an exploratory everyday-language check was run separately (LLM-graded, not in the repo). A human-judged evaluation is deferred; a blind human-judging harness was built and then removed as out of proportion for this scope. |
| Choice of ranking: semantic, keyword or combined (`--ranking`) | Deferred, evidence in hand | D18's two checks point in opposite directions by query wording; combining is never far from the best, but the gains are small, so the default waits for a larger, human-judged test. |
| "Find patents similar to this patent ID" | **Built (D20)** | `--similar-to`: the patent's abstract and first claim become the query, the patent itself is excluded, and all filters still apply. Covers the brief's example input "an entire patent (given by patent ID)". |
| Restricting search to claims or descriptions only | Deferred | Small; outside the first milestone. |
| Flagging identical abstracts as possibly related | Partly done (D20) | Flagged in similar-patent results only, worded as "possibly the same patent family"; identical text is not proof of one. |
| Links that open the patent at the matching passage | Deferred, if time allows | Planned with browser text fragments (`#:~:text=...`), one link per excerpt; needs a check that Google Patents' page text matches our passages. If the words aren't found, the link still opens the patent. |
| "Related by meaning" section after strict results | Deferred, if time allows | A separate, labelled list shown on expand: the same query re-run without the keyword filters (classification stays strict), skipping patents already shown. Unlike the rejected semantic filter (D11), strict results keep their meaning and no similarity cut-off is needed. |
| Feature-to-evidence mapping | Rejected for Part 1 | Several retrievals per query and complex aggregation, for a workflow not asked for. |
| Scraping patents not in the dataset | Rejected | A reliable local system matters more than an unreliable scraper; documented as an extension. |
| Process memory measurement | Skipped | Index sizes are reported (53 MB vectors, 44 MB metadata); readings on Windows add noise rather than insight. |

## D16. Show up to three matching pieces per result; keep ranking by the best piece
- **Status:** Validated (display); ranking change Deferred
- **Problem:** The first version showed only a patent's best-matching piece. A query often matches several claims or paragraphs of one patent, so the user didn't see them, and a patent with many strong matches looked the same as one with a single match.
- **Evidence:** Over the top 10 results of 12 test queries, a patent's second-best piece scored a median of 0.050 below its best (90th percentile 0.137), and its third-best 0.075 below. With a 0.10 margin, the second piece qualifies for 80% of results and the third for 66%; with 0.05, only 49% and 32%. Picking pieces naively would have shown the same evidence twice 18 times (another part of the same claim or paragraph, or identical text).
- **Alternatives:** Show every piece above a fixed score (rejected: scores aren't comparable across queries); show the top 3 regardless of score (rejected: weak third pieces are noise); rank patents by several pieces, for example the sum of their top 3 scores.
- **Chosen:** Show the best piece, then up to two more that score within 0.10 of it, skipping another part of a piece already shown and repeated text. Count the further pieces within the margin and show that count ("+17 more passages within 0.10 of the best match"). Ranking still uses the best piece only.
- **Rationale:** The count makes a patent's breadth of matches visible without changing the ranking. Changing how patents are ranked can only be justified by a relevance measurement; D18 measures overall ranking, not this choice.
- **Trade-offs:** Fetching evidence now scores each top patent's own pieces again: 0.4 to 0.7 ms per search on the laptop, up from about 0.05 ms, which raised the unfiltered median from 2.26 to 2.48 ms. Patents with many matches still don't rank higher.
- **Validation:** Unit tests on a hand-made index with known scores check the order, the margin, the cap, de-duplication, and that all filtering strategies give identical excerpts.
- **Limitations:** The 0.10 margin is tuned on 12 queries for this model; another model's scores would need a new measurement.

## D17. When filters leave nothing, suggest changes that would return patents
- **Status:** Validated
- **Problem:** The first version said which filter matched nothing and printed a general tip. A user with a typo, a near-miss title or an over-strict combination had to guess what to change.
- **Chosen:** On an empty result, the engine offers concrete changes, each with the number of patents it would give and a command that runs it. Only changes that would return patents are offered, at most six, in this order:
  1. **Spelling:** a keyword found nowhere in its field gets close words from that field's own vocabulary ("aerodinamic" → "aerodynamic", 7 patents).
  2. **Near titles:** an exact title with no match gets the closest real titles ("pneumatic tyre" → "PNEUMATIC TIRE", 33).
  3. **Nearby classification codes:** an unknown code gets the codes that exist under its closest existing prefix ("B60X" → B60C 318, B60B 298, B60D 20).
  4. **Any word instead of all:** when each word matches some patents but none matches them all, with the count per word ("flat" 4, "tyre" 294, "warning" 1 → 296 patents).
  5. **Search by meaning:** drop the filter and add its words to the query.
  6. **Drop one filter:** with the count that remains. When every filter matches something but no patent passes them all, each filter is offered for dropping.
- **Rationale:** Each suggestion is computed from the same indexes the filters use, so its count is exact, not a guess. Filters stay strict: nothing appears unless the user picks a suggestion.
- **Alternatives:** Relax filters automatically (rejected: results that don't satisfy what the user typed, shown without saying so); fuzzy matching inside the filters (still deferred, see D15).
- **Evidence:** Computing the suggestions takes 0.2 to 4.5 ms (median of 20 runs per case, 2-core sandbox), and only runs when a search returns nothing.
- **Validation:** Seven unit tests on the hand-made index: spelling, any-word, unknown classification, near title, moving words into the query, filters that never overlap, and no suggestions when there are results.
- **Limitations:** "Search by meaning" searches all of a patent's text, not just the field the filter named, because searching one field by meaning isn't built (D15). Spelling suggestions cover single words, not phrases. The near-title suggestion is a hint only; matching itself stays exact.

## D18. Measure search quality with two complementary checks; keep semantic ranking as the default
- **Status:** Validated (check 1, reproducible); check 2 exploratory; default ranking Provisional
- **Problem:** Speed was measured thoroughly, but search quality not at all. The choice of a semantic model over keyword search rested on an assumption.
- **History:** A blind, pooled evaluation was built first: both methods' top 5 for 10 queries, shuffled, with relevance rules written in advance, for a person to judge (about 80 judgments). Weighed against the take-home's roughly two-hour scope, it was out of proportion, so it was replaced by an automated check that needs no judging. Git history keeps the first version.
- **Check 1, self-retrieval (in the repo):** For each patent, its first claim is the query and that patent's own claims are hidden, so it can only be found through its abstract and description. The rank of the patent among all 640 is recorded for the semantic search, a BM25 keyword baseline (`src/bm25.py`, same passages, tokenizer, filters and best-passage ranking), and a combination of the two by reciprocal rank fusion (k = 60, the standard value, not tuned on this test). 26 patents with an identical twin are skipped: the twin's visible claims would be found first, which isn't a miss.
- **Check 1 evidence (`evaluation/self_retrieval.md`, 614 queries):**

  | Ranking | Recall@1 | Recall@10 | MRR |
  |---|---:|---:|---:|
  | Semantic | 52.4% | 82.2% | 0.627 |
  | BM25 | 86.5% | 93.8% | 0.893 |
  | Combined | 70.0% | 89.9% | 0.769 |

  Per query, semantic ranked the patent higher in 28 cases and BM25 in 257. In 344 patents the description repeats every distinctive word of the claim (Recall@1: semantic 58.1%, BM25 96.2%); in the 270 where some are missing, BM25 still won (45.2% against 74.1%).
- **Why a second check:** Check 1 favours keywords, since a patent's description usually repeats its claim's wording, and it can't test the case semantic search exists for: a user describing an idea in words different from the patent's. Rather than accept or dismiss the result on reasoning alone, that explanation was tested.
- **Check 2, paraphrase test (everyday-language queries, exploratory, run separately):** 12 technical information needs, each written twice: in everyday language and in wording close to the patents' terms (24 queries). Queries and grading rules were fixed before results were seen. The three methods' results were pooled (355 patents) and graded 0–2 for relevance, blind to method and rank. Metric: nDCG@10.

  | Query wording | Semantic | BM25 | Combined |
  |---|---:|---:|---:|
  | Everyday language | 0.475 | 0.335 | 0.468 |
  | Patent terminology | 0.836 | 0.807 | 0.844 |

  Semantic beat BM25 on 8 of the 12 everyday-language needs, most clearly where query and patent shared almost no words (wheel theft: first relevant patent at rank 2 against 118; self-inflating tyres: 4 against 45). The 95% interval for semantic's advantage on everyday language is about −0.001 to +0.30.
- **Interpretation:** The two checks agree rather than conflict. BM25 is strong when the query uses the patent's wording; semantic search is stronger when it doesn't. The general-purpose model (trained on web text, 384 dimensions, reading only the first 256 tokens of a query) is still weaker than BM25 at separating near-identical patents when the wording matches.
- **Decision:** Semantic search stays the default, because natural-language descriptions of an invention are the main use case and check 2 favours it there. Combining the two is never far from the best (nearly tied with semantic on everyday language, highest on patent terminology, between the two on check 1), so it is the next candidate for the default. The gains are small, so it waits for a larger, human-judged test.
- **Alternatives:** BM25 as the default (best on check 1, clearly worst on everyday language); combined as the default now (defensible, but the evidence for a gain is thin); a human-judged pooled evaluation (built, then removed for scope).
- **Trade-offs and limits:** Check 1 is a proxy, not prior-art relevance. Check 2 covers only 12 needs, and its interval only just touches zero, so it is suggestive, not conclusive. It was graded by an LLM (Claude) rather than people, and the same model helped write the queries; an LLM judges by meaning, which may favour semantic search. Grading covered only pooled results, so relevant patents no method found are missed. Its queries and grades are not in this repository, so it can't be reproduced from here. Both checks use one corpus of near-identical tire patents.
- **Validation:** Three unit tests on the check 1 ranking logic (own claims hidden, BM25 zero score is a miss, first claim chosen after cancelled ones); four on the BM25 baseline.

## D19. Cache word stemming
- **Status:** Validated
- **Problem:** Building the BM25 index took 47.7 s on the 2-core sandbox.
- **Evidence:** Almost all of it was stemming: about 3.35 million word occurrences, but only 19,979 distinct words.
- **Chosen:** Cache each word's stem (`functools.lru_cache` on `stem`).
- **Result:** 2.0 s, 24 times faster (3,348,406 cache hits, 19,979 misses). The same function serves the filter indexes and keyword parsing, so they benefit too. All tests still pass.
- **Trade-offs:** Memory for about 20,000 cached words, which is negligible.

## D20. Find patents similar to a given patent ID
- **Status:** Validated (behaviour and edge cases); relevance not separately measured
- **Problem:** The brief lists "an entire patent (given by patent ID)" as a possible input. A user who already holds one relevant patent wants its neighbours, without writing a query.
- **Chosen:** `--similar-to <ID>` (`PatentSearch.search_similar`). The query vector is built from the patent's stored vectors: the mean of its abstract pieces and the mean of its first claim's pieces, each normalised, then averaged with equal weight. It then runs through the normal search, so ranking by best passage, filters, the filtering-method choice and excerpts all work unchanged. The patent itself is excluded from its own results.
- **Rationale:** The abstract says what the invention is in plain words; the first claim says what is protected. Either alone misses half the picture. Using stored vectors means no model at query time: no 13-second cold start, and a search takes about 4 to 8 ms on the full sample.
- **Edge cases handled:**
  - **ID typed in another shape:** case, spaces and punctuation are ignored, and a leading `US` and trailing kind code (`A1`) are stripped, so `US20240051333A1`, `us 2024/0051333 a1` and `20240051333` all work.
  - **Unknown ID:** a clear message with up to three close IDs from the index, and exit code 2.
  - **No abstract:** the first claim alone is used; a patent with neither falls back to its description; a patent with no indexed text is refused with a message.
  - **Filters that match only the source patent:** a specific message ("the filters match only the source patent itself") instead of an empty result, and no suggestions, since no filter change would help.
  - **A query and `--similar-to` together:** refused; they describe two different searches.
  - **Identical abstracts:** a patent whose abstract is identical to the source's is kept, usually at the top (0.985 for the twin of 20240116313), and labelled "possibly the same patent family" rather than hidden (D13).
  - **Suggestions:** when filters match nothing, the commands shown keep `--similar-to`, and "search by meaning" is left out because it would need a text query.
- **Alternatives:** Use every passage of the patent (one long query, dominated by description paragraphs); score each candidate against all of the source's passages and aggregate (slower, and no clear rule for combining); hide identical-abstract patents (rejected: identical text is not proof of the same family).
- **Trade-offs:** The first claim is normally the main independent claim, but not always (it can follow cancelled claims). Only the first claim is used, so inventions whose other independent claims cover something different are represented partly.
- **Validation:** Five unit tests: the source is never returned and no model is loaded; IDs in common shapes resolve and a near miss gives close IDs; filters apply and all four filtering methods agree; filters matching only the source say so; a patent without an abstract uses its claim. Checked by hand on the full sample for the twin, an unknown ID, an exact-title filter matching only the source, a misspelled filter, and a query given together with `--similar-to`.

## D21. Sample the large dataset evenly, and load it defensively
- **Status:** Validated
- **Problem:** The large dataset is 471 batch files (about 471,000 patents, 30.7 GB); Part 3 allows a portion. File names carry no dates, and the first files alone could be unrepresentative.
- **Chosen:** `folder:N` takes N patents spread evenly across every `patents*.json` file (a fixed random sample within each file, seed 0). Patent IDs are de-duplicated across folders, so Part 1's data and the large sample can be indexed together.
- **Problems found while loading, and handled:**
  - A stray `sample.json` (a single record, not a batch list) was silently read as zero patents; the 300-patent index had 250. Only `patents*.json` files are read now, and a file that is not a list is reported.
  - 114 of the 300 sampled patents (38%) have the classification written as the text "N/A". It is now stored as missing, so no classification filter can match it.
- **Validation:** Five loading tests (files in subfolders found, a limit samples across the whole dataset, folders combined without duplicates, other JSON files ignored and bad batches reported, "N/A" treated as missing). The rebuilt index reports `classification_marked_missing 114`.

## D22. Profile before optimising: embedding is 99.5% of processing
- **Status:** Validated
- **Evidence:** `benchmarks/profile_large.py` on the large data: 99.5% of build time is embedding; the rest is loading and filter tables. Patents average 147 passages (median 107, 90th percentile 271), against 54 in Part 1, because descriptions are longer. At the laptop's rate (about 23 passages a second), all 471,000 patents would take about 26 days; 10^7 patents a year would need 47 passages a second around the clock.
- **Consequence:** By Amdahl's law, only the embedding step is worth speeding up, so every processing attempt (D23) targets it. Faster loading or filter tables could save at most 0.5%.

## D23. Keep PyTorch, batch 64 and one process: CPU speed-ups tried and rejected
- **Status:** Validated (the defaults); the alternatives Rejected
- **Method:** Each alternative was judged on a complete build from an empty cache (`benchmarks/build_ab.py`: a warm-up build, then A, B, B, A, averaged), with a check that the vectors are identical. Identical runs varied by up to 45% on the laptop, so smaller differences are treated as noise.
- **Results:**
  - **ONNX Runtime** (plain and optimised): slower than PyTorch at every batch size from 8 to 64 (`embed_speed.json`, `embed_speed_extra.json`).
  - **8-bit ONNX:** 8.4 to 8.6 of each top 10 the same as the current model, so it changes results; rejected without a relevance measurement to justify the trade.
  - **Batch 16:** 1.3–1.4× faster on 1,000 passages in random order, but 0.99× on a complete build (embedding 0.93×), because the build already sorts passages by length and batches have almost no padding to save (`build_ab_batch.json`). It was briefly the default; reverted when the complete build showed no gain.
  - **4 embedding processes** (`--workers 4`, one thread each): 0.85× on a complete build; embedding alone 1.19×, within noise. Each process loads its own model, and one PyTorch process already uses every core (`build_ab_workers.json`). The option stays, default 1.
- **Conclusion:** On a CPU, embedding is limited by raw compute. The lever is a GPU, costed in Part 2 at $31–188 on spot GPUs for a one-off embedding of every passage (`docs/PART2_DESIGN.md`, section 8.3). Not measured here, to keep Part 3 on the existing setup.

## D24. HNSW for searches without filters, with exact rescoring
- **Status:** Validated (up to 75,018 passages)
- **Problem:** Exact search scores every passage, so its time grows in step with the index (D9).
- **Chosen:** FAISS `IndexHNSWFlat` (`M=32`, `efConstruction=128`), built by `build_index` automatically at 50,000 vectors or more, in 12.7 s for 75,018. A search fetches 200 candidate passages, doubling until 10 distinct patents are found, then rescores the top 30 patents (3 × top 10) exactly against all their passages. A graph whose size no longer matches the index is detected and refused; a rebuild below the threshold deletes the old graph.
- **Why rescoring:** Without it, 97% of evidence passages matched exact search, because the graph returns only some of each patent's passages. With it, the score and evidence passage of each returned patent are exact.
- **Evidence** (940 patents, 75,018 passages, `search.json`): unfiltered retrieval 7.06 → 2.61 ms (2.7×) for Part 1's queries and 5.39 → 1.59 ms (3.4×) for 8 queries from other fields; recall@10 100% and 99% (minimum 90%); same evidence 100%. Complete query 1.4–1.5× faster, because embedding the query is unchanged. Scaling (`search_scaling.json`): 1.1× at 8,938 passages, 1.8× at 18,725, 3.3× at 37,562, 5.8× at 75,018.
- **Trade-offs:** Approximate: a query can lose a result (minimum recall 90%). The graph adds memory and build time, and must be rebuilt when the index changes.

## D25. Searches with filters stay exact
- **Status:** Validated; the rule changed twice after measurements
- **History:** The first rule reused Part 1's 8% threshold for HNSW and was wrong: HNSW was twice as slow at 20%. The second used HNSW when filters kept at least half the patents. Queries from other fields then showed recall of 88% (minimum 60%) even with a broad filter keeping 68% of patents, and as low as 33% with a narrow one, for at most a 1.5× gain.
- **Chosen:** HNSW only when no filter is set (a similar-patent search, which only excludes the source patent, still uses it). Any filter uses Part 1's prefilter or mask.
- **Evidence:** With this rule, every filtered scenario has 100% recall and runs at the same speed as before (0.8–1.2×) (`search.json`).
- **Trade-off:** Filtered search still grows linearly with the index. At Part 2's scale, filters run inside OpenSearch's vector search instead.

## Later experiments (not yet run)

Part 3 ran several of these on the large dataset: ONNX, int8, worker processes (D23) and HNSW
(D24). The rest are still untried.

Ideas for reducing each measured cost, ordered by the cost they target. None has been tried, so
there are no predicted numbers: each would be measured with `benchmarks/timing.py` (and
`embedding_run.json` for indexing). Anything that changes the vectors also changes the results,
so it can only be adopted together with a relevance measurement; until then, it can be judged
on speed and on how much the top-10 results change.

**Query embedding (median 8.1 ms, the largest per-search cost) and model load (12.8 s)**

| Experiment | Why it might help | Measure | Risk |
|---|---|---|---|
| Run the same model through ONNX Runtime or OpenVINO (`SentenceTransformer(..., backend="onnx")`) | Leaner CPU inference than PyTorch for one short text | Embedding median and p95; model load time | Vectors differ slightly: re-embed the corpus with the same backend |
| Int8-quantized version of the model (ONNX dynamic quantization) | Smaller weights, faster arithmetic on CPU | Embedding time; overlap of top-10 results with the current model | Some loss of quality |
| A static embedding model (for example `static-retrieval-mrl-en-v1`, or a Model2Vec distillation) | No transformer at query time: a lookup and an average | Embedding time; top-10 overlap | Clearly weaker on paraphrases; needs the relevance evaluation |
| Cache embeddings of repeated queries | A repeated query skips the model entirely | Hit rate on a realistic query log | Only helps repeated queries |
| Keep the model loaded in a long-running process (a Streamlit app or a small API) | The CLI pays the 12.8 s load on every call | Time from request to first result | None for results; it is an architecture change |
| Tune PyTorch's thread count for single queries | Thread start-up can dominate a tiny batch | Embedding median and p95 | Machine-dependent |

**Index build (20 to 29 pieces per second on the laptop)**

| Experiment | Why it might help | Measure | Risk |
|---|---|---|---|
| ONNX, OpenVINO or int8 backends for bulk encoding | Same reasons as above, over 32,000 texts | Pieces per second | Same as above |
| Larger batches, or several worker processes (`start_multi_process_pool`) | Better use of the 8 cores | Pieces per second | Memory use |
| A GPU | Embedding is parallel arithmetic | Pieces per second | Not available on the laptop; matters at scale (Part 2) |

**Search after embedding (median 2.48 ms: scoring 1.87, ranking 0.17, fetching evidence 0.46)**

| Experiment | Why it might help | Measure | Risk |
|---|---|---|---|
| Reuse the scores from the main pass when fetching evidence, instead of scoring each top patent's pieces again | The mask and no-filter paths already hold every score; this is the 0.4 to 0.7 ms added by D16 | Evidence time, which should fall back towards 0.05 ms on those paths | None for results: same scores |
| FAISS exact index (`IndexFlatIP`) instead of NumPy | Tuned kernels for the same exact search | Scoring time; results must be identical | Small gain expected at 35,000 vectors |
| Compressed vectors: FAISS scalar quantization (8-bit), or binary embeddings with rescoring of the top candidates | Less memory to read per query: a quarter, or a thirty-second | Scoring time; top-10 overlap | Some loss of accuracy; mainly a large-scale win |
| Fewer dimensions with a model trained for it (Matryoshka embeddings) | 128 instead of 384 numbers per vector | Scoring time; top-10 overlap | MiniLM isn't trained for truncation; needs a different model |
| Approximate search (HNSW) | Avoids scoring every vector | Scoring time; recall against exact search | Not worth it at this size; essential at 10 million patents |
| Calibrate the prefilter/mask threshold automatically at start-up | The 8% crossover moved between machines | Whether `auto` picks the faster method on a new machine | A short start-up cost |

**Filter lookups (0.005 to 0.15 ms)**

Already about 6% of a search or less, so not worth optimizing at this size. At 10 million
patents, Python sets would be replaced by compressed bitmaps (for example Roaring bitmaps) or
by the database's own indexes, where intersections are much cheaper (Part 2).

## Caveat on what this system can conclude

The sample is 640 vehicle-related US patent applications published between February 2024 and
May 2025. It cannot support prior-art, novelty or infringement conclusions: real prior art
includes older patents, other jurisdictions, granted patents and non-patent literature. The
engine itself is domain-independent; the conclusions are limited by the data.
