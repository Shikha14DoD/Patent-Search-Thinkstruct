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

Decisions D1 to D7 concern the data; D8 to D14 and D16 onwards concern search; D15 records scope.

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
- **Interview explanation:** "Half the description paragraphs in the data are empty strings, and 119 patents have no description at all. I kept every patent and just skipped the blanks, because dropping a fifth of the corpus would hide patents whose claims could be exactly what the user needs."

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
- **Interview explanation:** "A third of the claims had lost their number in conversion, and some patents start with cancelled claims, so counting list positions put every label off. I rebuilt the numbering from the printed numbers and the cancelled ranges, flagged the inferred ones, and then checked the result against the patents' own cross-references: 96.7% of 'according to claim N' citations land on a claim we produced, including 4,300 on numbers we inferred."

### D3. The weekly date is the publication date, not the filing date
- **Status:** Validated (revised: the first version called it `filing_date`)
- **Problem:** Date filters and displays must say what the date actually is.
- **Evidence:** File names such as `US20240051333A1-20240215.XML`: A1 marks a pre-grant publication, and the date is when it was published. Applications usually publish about 18 months after filing.
- **Chosen:** Store and display it as `publication_date`.
- **Trade-offs:** No filing date is available in this data.
- **Interview explanation:** "I'd first labelled it a filing date. The A1 kind code says it's a publication, which comes about 18 months after filing, so calling it a filing date would have made any date-based reasoning wrong."

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
- **Chosen:** Each unit stores the embedded text, the displayed text and the raw source text. Notes mark reconstructed or added content: "claim number inferred", "opening text missing in source", and "matched together with Claim N, which it depends on".

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
- **Interview explanation:** "I'd been splitting text by word count and assumed it fit the model. When I measured with the real tokenizer, 17% of dependent claims were losing the end of their own text, because I'd prepended the parent claim. Now the claim's own words always fit, the parent context is trimmed to whatever space is left, and the build verifies that zero units exceed the limit."

---

## Search

### D8. Search passages, rank each patent by its best passage
- **Status:** Validated for evidence; relevance not yet measured
- **Problem:** Choose the unit that is embedded and compared with the query.
- **Evidence:** A patent with a description has a median of 4,195 description tokens (90th percentile 10,303). Only 1.7% of whole patents would fit in 512 tokens, none in 256. By contrast, a single claim has a median of 42 tokens, a description chunk 103 and an abstract 147.
- **Alternatives:** (A) whole patents: the model would see only the first 256 tokens, about 6% of a typical patent. (B) Patent-level retrieval, then evidence extraction: smaller index, but the first stage can miss a patent whose only relevant passage is a single claim. (C) Separate abstract and passage indexes: duplication and more complex ranking, without a measured need.
- **Chosen:** Index abstracts, claims and description chunks (34,551 units, 54 per patent). A patent's score is its best unit's score, and that unit is the evidence shown.
- **Rationale:** It is the only option where the evidence shown is the text that actually matched, and no part of a patent is invisible to the search.
- **Trade-offs:** 54 vectors per patent instead of one. At 10 million patents that is roughly 540 million vectors, about 830 GB as 384-dimensional float32, so Part 2 needs approximate search and compression. Long patents get more chances at a high-scoring passage.
- **Validation still needed:** A relevance evaluation (see D14 and D15).
- **Interview explanation:** "A typical patent here is over 4,000 tokens and the model reads 256, so embedding whole patents would mean judging each one by its first page. Searching passages means every claim and paragraph is searchable, and the result can show exactly which claim matched. The price is 54 vectors per patent, which matters at 10 million patents, and that's a Part 2 problem I can size precisely."

### D9. all-MiniLM-L6-v2 with exact search in NumPy
- **Status:** Provisional (model); Validated (exact search at this scale)
- **Problem:** Choose an embedding model and a vector search method.
- **Evidence:** Exact search over all 34,551 vectors takes a median of 2.26 ms; embedding a query takes a median of 11.4 ms. The vectors take 53 MB. Building the embeddings ran at 20 to 29 texts per second on the laptop's CPU, about 20 minutes for the full sample.
- **Alternatives:** BGE-small (same size, often stronger on retrieval benchmarks); patent-specific models; FAISS or another approximate index.
- **Chosen:** all-MiniLM-L6-v2: small, fast on CPU, widely used, and quick for a reviewer to reproduce. Exact search with one matrix multiplication.
- **Rationale:** At 2.26 ms, an approximate index would add infrastructure and a recall loss to save almost nothing, since query embedding dominates the request.
- **Trade-offs:** The model isn't tuned for patent language. Exact search grows linearly; around a million vectors it would reach tens of milliseconds.
- **Not done, and why:** Comparing models requires a relevance metric. Without an evaluation set, a model swap would be guesswork, so the comparison is deferred together with evaluation.

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
- **Evidence:** Resolving filters takes 0.004 to 0.12 ms in every benchmark scenario.
- **Alternatives:** A semantic fallback for filters (return "related" patents when no exact match exists). Rejected: a filter is a yes/no decision, and similarity has no natural cut-off, so any threshold would be arbitrary and the results hard to explain. Meaning-based search belongs in the ranked query.
- **Limitations:** Keyword filters don't match synonyms ("car" vs "vehicle"). Classification codes are stored without separators (`B60B2700FI`), so prefixes deeper than the subclass can be ambiguous. Generic titles are common: 33 patents are titled exactly "PNEUMATIC TIRE".

### D12. Choose the filtering method by how many units survive
- **Status:** Validated (revised twice)
- **Problem:** The brief says naive hybrid search is expensive and asks for timings with and without it.
- **History:** The plan assumed that filtering first and scoring only the survivors (prefilter) would always be faster. The first single runs showed a filtered search slower than an unfiltered one, so the benchmark compared three methods: prefilter (copy out and score only surviving units), mask (score all units in place, discard failures) and postfilter (the naive version: rank everything, then check filters by reading text). A provisional threshold of 25% came from a 2-core sandbox; the laptop measurements moved it to 8%.
- **Evidence (laptop, median ms after the query is embedded):**

  | Surviving units | Prefilter | Mask | Postfilter (naive) |
  |---|---:|---:|---:|
  | 100% (no filter) | – | 2.26 | – |
  | 44.2% (B60B) | 8.92 | 2.22 | 5.44 |
  | 19.1% (B60C11) | 3.99 | 2.10 | 5.35 |
  | 9.5% (abstract: tread groove) | 2.22 | 2.09 | 12.07 |
  | 6.4% (exact title) | 1.61 | 2.08 | 6.49 |
  | 5.2% (title/abstract: sensor) | 1.49 | 2.32 | 18.17 |

  The naive method is 2 to 8 times slower than no filter. Mask costs the same as no filter in every case. Prefilter wins only when a small share survives: copying out thousands of scattered vectors costs more than one multi-threaded multiplication over the whole contiguous matrix. All three return identical results in every case.
- **Chosen:** `auto`: prefilter when at most 8% of units survive, mask otherwise. 8% lies between the laptop's crossover (between 6.4% and 9.5%) and stays safe on the 2-core machine, where prefilter kept winning up to about 20%.
- **Trade-offs:** The crossover depends on hardware, so 8% is a tuned value, not a property of the method.
- **Implication for scale:** At this size, filtering can't make search much faster; it can only avoid making it slower. Its real value appears at large scale, where an index partitioned by classification would let a filter skip whole partitions without copying (Part 2).
- **Interview explanation:** "I expected pre-filtering to always win, and measured three methods instead of assuming. The naive way, ranking first and filtering after, was 2 to 8 times slower. But pre-filtering only won below about 8% of the data surviving; above that, copying the surviving vectors cost more than scoring everything in place. So the engine picks the method from the filter's selectivity, and that threshold came from measurements on two machines, which also showed it moves with hardware."

### D13. Break score ties deterministically
- **Status:** Validated
- **Problem:** The same query must give the same results whichever filtering method runs it.
- **Evidence:** Identical passages appear in several patents (11 groups of identical abstracts covering 26 patents; 416 paragraphs appear in three or more patents), so those patents get exactly the same score. The benchmark's consistency check found 6 of 84 query-filter combinations where methods disagreed on the 10th result.
- **Chosen:** Keep every patent tied with the k-th score, then break ties by index position.
- **Validation:** All 84 filtered query-scenario combinations (12 queries x 7 filter scenarios) now agree across methods.
- **Related:** Patents with identical abstracts are not merged; identical text doesn't prove they are the same patent family.

### D14. Semantic ranking only for now; keyword (BM25) fusion deferred
- **Status:** Deferred
- **Problem:** Embeddings match meaning; keyword ranking matches exact terms. Combining them may help.
- **Evidence:** "wheel rim" is the kind of query where keyword ranking helps: the query's exact words are a patent's title. Semantic search still ranks that patent first after D6, but by a small margin (0.703 vs 0.698).
- **Why deferred:** Without a relevance metric there is no way to show whether fusion helps or hurts, and fusion adds a tuning parameter. A measured BM25 baseline belongs with the evaluation work.

---

## D15. Scope: what was deliberately not built

| Item | Decision | Reason |
|---|---|---|
| Fuzzy title matching | Deferred | Exact normalised matching meets the brief; fuzzy matching needs tuning and a test set. |
| Million-vector synthetic benchmark | Deferred | Part 2 or Part 3; the real crossover data above already answers the brief's question at this size. |
| Relevance evaluation (self-claim retrieval with leakage controls, hand-reviewed queries) | Deferred | Needed before BM25 or model comparisons; a candidate for Part 3. |
| "Find patents similar to this patent ID" | **Planned: will be built** | Input a patent number from the dataset; its abstract and independent claims become the query, the patent itself is excluded from the results, and all filters still apply. Covers the brief's example input "an entire patent (given by patent ID)". |
| Restricting search to claims or descriptions only | Deferred | Small; outside the first milestone. |
| Flagging identical abstracts as possibly related | Deferred | Needs a careful label; identical text is not proof of a patent family. |
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
- **Rationale:** The count makes a patent's breadth of matches visible without changing the ranking. Changing how patents are ranked can only be justified by a relevance measurement, which doesn't exist yet.
- **Trade-offs:** Fetching evidence now scores each top patent's own pieces again: about 0.3 ms per search instead of 0.05 ms, measured on the 2-core sandbox. Patents with many matches still don't rank higher.
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

## Caveat on what this system can conclude

The sample is 640 vehicle-related US patent applications published between February 2024 and
May 2025. It cannot support prior-art, novelty or infringement conclusions: real prior art
includes older patents, other jurisdictions, granted patents and non-patent literature. The
engine itself is domain-independent; the conclusions are limited by the data.