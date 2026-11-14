"""The search engine: filters narrow, the query ranks.

A query is embedded once and compared with the stored unit vectors. Each
patent is ranked by its best-matching unit (abstract, claim or description
paragraph), and that unit is returned as the evidence for the match.

Filters are strict: a patent that fails one never appears. Three ways of
applying them are implemented so the benchmark can compare them. All three
return the same results; only the work done differs.

  prefilter   Resolve the filters first through the inverted indexes, then
              copy out and score only the units of the surviving patents.
              Cheap when few patents survive.
  mask        Resolve the filters first, score every unit in place, then
              discard the patents that failed. No copying, so it is the
              better choice when most patents survive.
  postfilter  The naive version: score and rank everything, then walk down
              the ranking checking each patent against the filters by
              reading its text, with no index. Kept only as a baseline.
  auto        Pick prefilter or mask from the share of units that survive.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field

import numpy as np

from .text_utils import normalize_title, parse_keywords

STRATEGIES = ("prefilter", "mask", "postfilter", "auto")
# "auto" copies out the surviving units only when they are at most this
# share of the index. Provisional: to be set from benchmarks/results.md.
AUTO_PREFILTER_MAX_SHARE = 0.25


@dataclass
class Filters:
    """Everything the user can constrain. All of them are optional."""

    classification: str = ""      # "B60B" or "B60B, B60C11" (prefixes, OR-ed)
    title_keywords: str = ""      # words, or "a phrase" in quotes
    abstract_keywords: str = ""
    title_or_abstract_keywords: str = ""
    exact_title: str = ""
    match_any: bool = False       # False = every keyword must appear

    def active(self) -> bool:
        return any([self.classification, self.title_keywords, self.abstract_keywords,
                    self.title_or_abstract_keywords, self.exact_title])


@dataclass
class Result:
    rank: int
    patent_id: str
    title: str
    classification: str
    publication_date: str
    url: str
    score: float
    excerpt_label: str   # "Claim 4", "Description para 12", "Abstract"
    excerpt_note: str    # caveats about the excerpt, e.g. "claim number inferred"
    excerpt_text: str


@dataclass
class SearchReport:
    """Results plus the numbers the benchmark and the UI need."""

    results: list[Result]
    filter_counts: dict[str, int] = field(default_factory=dict)
    candidate_patents: int = 0
    candidate_units: int = 0   # units whose score was actually used
    scored_units: int = 0      # units that were multiplied with the query
    total_units: int = 0
    strategy: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    empty_reason: str = ""


class PatentSearch:
    def __init__(self, index_dir: str = "index", model=None):
        with open(f"{index_dir}/index.pkl", "rb") as handle:
            bundle = pickle.load(handle)
        self.vectors: np.ndarray = np.load(f"{index_dir}/vectors.npy")
        self.patents: dict = bundle["patents"]
        self.units: list[dict] = bundle["units"]
        self.title_index = bundle["title_index"]
        self.abstract_index = bundle["abstract_index"]
        self.title_or_abstract_index = bundle["title_or_abstract_index"]
        # Padded with spaces so a phrase can only match on word boundaries.
        self.title_phrases = {pid: f" {text} " for pid, text in bundle["title_phrases"].items()}
        self.abstract_phrases = {pid: f" {text} "
                                 for pid, text in bundle["abstract_phrases"].items()}
        self.exact_title_index = bundle["exact_title_index"]
        self.classification_prefix = bundle["classification_prefix"]
        self.model_name = bundle["model_name"]
        self._model = model

        # Every patent owns one contiguous block of unit rows. Keeping the
        # block bounds as arrays lets ranking work on whole blocks at once.
        unit_rows: dict[str, tuple[int, int]] = bundle["unit_rows"]
        self.patent_order: list[str] = sorted(unit_rows, key=lambda pid: unit_rows[pid][0])
        self.patent_position = {pid: i for i, pid in enumerate(self.patent_order)}
        self.block_starts = np.array([unit_rows[pid][0] for pid in self.patent_order],
                                     dtype=np.int64)
        self.block_ends = np.array([unit_rows[pid][1] for pid in self.patent_order],
                                   dtype=np.int64)
        self.unit_patent = np.repeat(np.arange(len(self.patent_order)),
                                     self.block_ends - self.block_starts)

    # -- model ------------------------------------------------------------
    @property
    def model(self):
        """Loaded on first use so that filter-only searches need no model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True,
                                 normalize_embeddings=True).astype(np.float32)[0]

    # -- filters, resolved through the indexes ----------------------------
    @staticmethod
    def _combine(matched: set[str] | None, hits: set[str], match_any: bool) -> set[str]:
        if matched is None:
            return hits
        return matched | hits if match_any else matched & hits

    def _keyword_match(self, terms, inverted, phrase_sources, match_any: bool) -> set[str]:
        """Patents matching the given terms in one field (or pair of fields)."""
        matched: set[str] | None = None
        for _, tokens in terms:
            if len(tokens) > 1:
                # Phrase: the words must appear next to each other.
                needle = f" {' '.join(tokens)} "
                hits = {pid for source in phrase_sources
                        for pid, text in source.items() if needle in text}
            else:
                hits = set(inverted.get(tokens[0], ()))
            matched = self._combine(matched, hits, match_any)
        return matched if matched is not None else set()

    def resolve_filters(self, filters: Filters) -> tuple[set[str] | None, dict[str, int], str]:
        """Return (allowed patent ids, per-filter counts, name of an empty filter).

        None means "no filter applied", which is different from an empty set
        ("filters applied and nothing matched").
        """
        counts: dict[str, int] = {}
        allowed: set[str] | None = None
        empty_reason = ""

        def apply(name: str, hits: set[str]) -> None:
            nonlocal allowed, empty_reason
            counts[name] = len(hits)
            if not hits and not empty_reason:
                empty_reason = name
            allowed = hits if allowed is None else allowed & hits

        if filters.classification:
            codes = self._classification_codes(filters.classification)
            hits: set[str] = set()
            for code in codes:  # several codes are OR-ed together
                hits |= self.classification_prefix.get(code, set())
            apply(f"classification {'/'.join(codes)}", hits)

        if filters.exact_title:
            key = normalize_title(filters.exact_title)
            apply(f'title is "{filters.exact_title}"',
                  set(self.exact_title_index.get(key, ())))

        if filters.title_keywords:
            apply(f'title "{filters.title_keywords}"', self._keyword_match(
                parse_keywords(filters.title_keywords), self.title_index,
                [self.title_phrases], filters.match_any))

        if filters.abstract_keywords:
            apply(f'abstract "{filters.abstract_keywords}"', self._keyword_match(
                parse_keywords(filters.abstract_keywords), self.abstract_index,
                [self.abstract_phrases], filters.match_any))

        if filters.title_or_abstract_keywords:
            apply(f'title/abstract "{filters.title_or_abstract_keywords}"', self._keyword_match(
                parse_keywords(filters.title_or_abstract_keywords),
                self.title_or_abstract_index,
                [self.title_phrases, self.abstract_phrases], filters.match_any))

        if allowed is not None:
            counts["combined"] = len(allowed)
        return allowed, counts, empty_reason

    @staticmethod
    def _classification_codes(raw: str) -> list[str]:
        return [code.strip().upper() for code in raw.replace(",", " ").split() if code.strip()]

    # -- filters, checked without any index (naive baseline only) ---------
    def _passes_without_index(self, patent_id: str, filters: Filters) -> bool:
        """Evaluate the filters on one patent by reading its text."""
        patent = self.patents[patent_id]

        def field_matches(raw: str, sources: list[str]) -> bool:
            outcomes = []
            for _, tokens in parse_keywords(raw):
                needle = f" {' '.join(tokens)} "
                outcomes.append(any(needle in source for source in sources))
            if not outcomes:
                return False
            return any(outcomes) if filters.match_any else all(outcomes)

        if filters.classification:
            code = patent["classification"].upper()
            if not any(code.startswith(prefix)
                       for prefix in self._classification_codes(filters.classification)):
                return False
        if filters.exact_title:
            if normalize_title(patent["title"]) != normalize_title(filters.exact_title):
                return False
        title = self.title_phrases.get(patent_id, " ")
        abstract = self.abstract_phrases.get(patent_id, " ")
        if filters.title_keywords and not field_matches(filters.title_keywords, [title]):
            return False
        if filters.abstract_keywords and not field_matches(filters.abstract_keywords, [abstract]):
            return False
        if filters.title_or_abstract_keywords and not field_matches(
                filters.title_or_abstract_keywords, [title, abstract]):
            return False
        return True

    # -- ranking ----------------------------------------------------------
    @staticmethod
    def _top_k(best: np.ndarray, top_k: int) -> np.ndarray:
        """Positions of the top_k highest values, best first, ties by position."""
        usable = np.flatnonzero(np.isfinite(best))
        if usable.size > top_k:
            keep = np.argpartition(-best[usable], top_k - 1)[:top_k]
            usable = usable[keep]
        return usable[np.lexsort((usable, -best[usable]))]

    def _rank_all_rows(self, scores: np.ndarray, allowed_mask: np.ndarray | None,
                       top_k: int) -> list[tuple[str, float, int]]:
        """Best unit per patent from a full score vector, optionally masked."""
        best = np.maximum.reduceat(scores, self.block_starts)
        if allowed_mask is not None:
            best = np.where(allowed_mask, best, -np.inf)
        ranked = []
        for position in self._top_k(best, top_k):
            start, end = self.block_starts[position], self.block_ends[position]
            row = int(start + np.argmax(scores[start:end]))
            ranked.append((self.patent_order[position], float(best[position]), row))
        return ranked

    def _rank_gathered(self, query_vector: np.ndarray, allowed_mask: np.ndarray,
                       top_k: int) -> tuple[list[tuple[str, float, int]], int]:
        """Copy out the surviving patents' units and rank only those."""
        positions = np.flatnonzero(allowed_mask)
        rows = np.flatnonzero(allowed_mask[self.unit_patent])
        scores = self.vectors[rows] @ query_vector
        lengths = self.block_ends[positions] - self.block_starts[positions]
        sub_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        best = np.maximum.reduceat(scores, sub_starts)
        ranked = []
        for local in self._top_k(best, top_k):
            start, end = sub_starts[local], sub_starts[local] + lengths[local]
            row = int(rows[start + np.argmax(scores[start:end])])
            ranked.append((self.patent_order[positions[local]], float(best[local]), row))
        return ranked, int(rows.size)

    def _rank_postfilter(self, scores: np.ndarray, filters: Filters,
                         top_k: int) -> list[tuple[str, float, int]]:
        """Naive baseline: rank every unit, then filter while walking down."""
        decided: dict[str, bool] = {}
        ranked: list[tuple[str, float, int]] = []
        for row in np.argsort(-scores, kind="stable"):
            patent_id = self.patent_order[self.unit_patent[row]]
            if patent_id not in decided:
                decided[patent_id] = self._passes_without_index(patent_id, filters)
                if decided[patent_id]:
                    ranked.append((patent_id, float(scores[row]), int(row)))
                    if len(ranked) >= top_k:
                        break
        return ranked

    # -- search -----------------------------------------------------------
    def search(self, query: str = "", filters: Filters | None = None, top_k: int = 10,
               strategy: str = "prefilter",
               query_vector: np.ndarray | None = None) -> SearchReport:
        """Run one search.

        query_vector lets a caller that already embedded the query (the
        benchmark) skip the embedding step.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        filters = filters or Filters()
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        report = SearchReport(results=[], total_units=len(self.units), timings_ms=timings)
        has_query = bool(query) or query_vector is not None
        naive = strategy == "postfilter" and filters.active() and has_query

        # 1. Filters. The naive baseline skips this and filters after ranking.
        allowed: set[str] | None = None
        if not naive:
            start = time.perf_counter()
            allowed, report.filter_counts, empty_reason = self.resolve_filters(filters)
            timings["filter_ms"] = (time.perf_counter() - start) * 1000
            if allowed is not None and not allowed:
                report.empty_reason = empty_reason
                timings["total_ms"] = (time.perf_counter() - total_start) * 1000
                return report
        report.candidate_patents = len(allowed) if allowed is not None else len(self.patents)

        # 2. No query: nothing to rank by, so list the newest publications.
        if not has_query:
            candidates = allowed if allowed is not None else set(self.patents)
            newest = sorted(candidates, key=lambda pid: (
                self.patents[pid]["publication_date"], pid), reverse=True)[:top_k]
            report.results = [self._as_result(rank, pid, 0.0, None)
                              for rank, pid in enumerate(newest, start=1)]
            report.strategy = "filter-only"
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000
            return report

        # 3. Embed the query.
        if query_vector is None:
            start = time.perf_counter()
            query_vector = self.embed_query(query)
            timings["embed_ms"] = (time.perf_counter() - start) * 1000

        # 4. Score units and keep each patent's best one.
        allowed_mask = None
        if allowed is not None:
            allowed_mask = np.zeros(len(self.patent_order), dtype=bool)
            allowed_mask[[self.patent_position[pid] for pid in allowed
                          if pid in self.patent_position]] = True
            report.candidate_units = int(
                (self.block_ends - self.block_starts)[allowed_mask].sum())
        else:
            report.candidate_units = len(self.units)

        if strategy == "auto":
            share = report.candidate_units / max(len(self.units), 1)
            strategy = ("prefilter" if allowed is not None
                        and share <= AUTO_PREFILTER_MAX_SHARE else "mask")

        if naive:
            start = time.perf_counter()
            scores = self.vectors @ query_vector
            timings["score_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            ranked = self._rank_postfilter(scores, filters, top_k)
            timings["rank_ms"] = (time.perf_counter() - start) * 1000
            report.scored_units = len(self.units)
            report.candidate_units = len(self.units)
            report.candidate_patents = len(self.patents)
            report.strategy = "postfilter"
        elif strategy == "prefilter" and allowed_mask is not None:
            start = time.perf_counter()
            ranked, report.scored_units = self._rank_gathered(query_vector, allowed_mask, top_k)
            # Copying, scoring and ranking are interleaved here, so they are
            # reported together.
            timings["score_ms"] = (time.perf_counter() - start) * 1000
            timings["rank_ms"] = 0.0
            report.strategy = "prefilter"
        else:
            start = time.perf_counter()
            scores = self.vectors @ query_vector
            timings["score_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            ranked = self._rank_all_rows(scores, allowed_mask, top_k)
            timings["rank_ms"] = (time.perf_counter() - start) * 1000
            report.scored_units = len(self.units)
            report.strategy = "mask" if allowed_mask is not None else "no-filter"

        # 5. Fetch the evidence for the winners.
        start = time.perf_counter()
        report.results = [self._as_result(rank, patent_id, score, row)
                          for rank, (patent_id, score, row) in enumerate(ranked, start=1)]
        timings["evidence_ms"] = (time.perf_counter() - start) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        return report

    def _as_result(self, rank: int, patent_id: str, score: float, row: int | None) -> Result:
        patent = self.patents[patent_id]
        if row is None:
            label, note, text = "Abstract", "", patent["abstract"]
        else:
            unit = self.units[row]
            notes = [unit["label_note"]] if unit.get("label_note") else []
            if unit.get("context_label"):
                notes.append(f"matched together with {unit['context_label']}, "
                             "which it depends on")
            label, note, text = unit["label"], "; ".join(notes), unit["display_text"]
        return Result(rank=rank, patent_id=patent_id, title=patent["title"],
                      classification=patent["classification"],
                      publication_date=patent["publication_date"], url=patent["url"],
                      score=round(score, 4), excerpt_label=label,
                      excerpt_note=note, excerpt_text=text)