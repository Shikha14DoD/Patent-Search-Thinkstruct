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
  hnsw        Approximate search with an HNSW graph (FAISS), for large indexes.
              Built by build_index when an index has 50,000 vectors or more.
              Filters are applied inside the graph search, and every returned
              patent is checked against them again.
  exact       Part 1's choice between prefilter and mask; never HNSW.
  auto        HNSW when the index has one and no filter is set; otherwise
              prefilter or mask, from the share of units that survive.
              Measured in benchmarks/large/search.json.
"""

from __future__ import annotations

import bisect
import difflib
import os
import pickle
import re
import time
from dataclasses import dataclass, field, replace

import numpy as np

from .text_utils import TOKEN_RE, normalize_title, parse_keywords, stem

STRATEGIES = ("prefilter", "mask", "postfilter", "hnsw", "exact", "auto")
# A result shows up to this many matching pieces of its patent.
MAX_EXCERPTS = 3
# An extra piece is shown only if it scores within this much of the
# patent's best piece. Measured on 12 test queries: the 2nd-best piece is a
# median 0.05 below the best, the 3rd-best 0.075; a 0.10 margin keeps the
# 2nd for 80% of top-10 results and the 3rd for 66% (see DECISIONS.md D16).
EXCERPT_MARGIN = 0.10
# "auto" copies out the surviving units only when they are at most this
# share of the index. Set from benchmarks/results.md: on an 8-core laptop
# prefilter beat mask at 5-6% of units and lost from about 9% upwards; on a
# 2-core machine it kept winning up to about 20%. 8% is safe on both. The
# crossover depends on hardware, so this is a tuned value, not a constant
# of the method (see DECISIONS.md).
AUTO_PREFILTER_MAX_SHARE = 0.08
# HNSW, when available, is used only without filters (a similar-patent
# search, which only excludes the source patent, still uses it). Measured on 75,018 passages (benchmarks/large/search.json): unfiltered, the
# same top 10 as exact search for Part 1's queries and 99% for queries from
# other fields; with any filter, recall fell (88% for other-field queries even
# with a broad filter keeping 68% of patents, 64% with a narrow one) for at
# most a 1.5x gain, because a filtered graph search misses allowed passages.
# Passages fetched from the graph before grouping by patent; also its search
# width. 200 gave 100% recall@10 without filters (100 gave 98%).
HNSW_CANDIDATES = 200
HNSW_FILE = "hnsw.faiss"
# Candidate patents (best first, by the graph) rescored exactly over all their
# passages, as a multiple of the results asked for. Rescoring every patent the
# graph touched halved the speed-up; 3x kept recall and evidence identical.
HNSW_RESCORE_FACTOR = 3


@dataclass
class Filters:
    """Everything the user can constrain. All of them are optional."""

    classification: str = ""      # "B60B" or "B60B, B60C11" (prefixes, OR-ed)
    title_keywords: str = ""      # words, or "a phrase" in quotes
    abstract_keywords: str = ""
    title_or_abstract_keywords: str = ""
    exact_title: str = ""
    match_any: bool = False       # False = every keyword must appear
    # Publication date range, inclusive: "2024", "2024-03" or "2024-03-15". The
    # data has publication dates only (the weekly file names, D3), not filing dates.
    published_from: str = ""
    published_to: str = ""

    def active(self) -> bool:
        return any([self.classification, self.title_keywords, self.abstract_keywords,
                    self.title_or_abstract_keywords, self.exact_title,
                    self.published_from, self.published_to])


DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def date_bound(text: str, end: bool) -> str:
    """'2024' -> '2024-01-01' (from) or '2024-12-31' (to); '2024-03' -> '2024-03-01' / '2024-03-31'.

    Bounds are compared as text, which orders ISO dates correctly; '-31' is a
    safe end for any month because no date sorts after it within that month.
    """
    text = text.strip()
    if not DATE_RE.match(text):
        raise ValueError(f"date {text!r} is not YYYY, YYYY-MM or YYYY-MM-DD")
    if len(text) == 4:
        return text + ("-12-31" if end else "-01-01")
    if len(text) == 7:
        return text + ("-31" if end else "-01")
    return text


def newest_first(results: list[Result]) -> list[Result]:
    """The same results, newest publication first (ties keep their relevance order).

    This reorders the top results; it does not fetch the newest patents overall.
    """
    ordered = sorted(results, key=lambda r: r.publication_date or "", reverse=True)
    return [replace(result, rank=rank) for rank, result in enumerate(ordered, start=1)]


@dataclass
class Excerpt:
    label: str   # "Claim 4", "Description para 12", "Abstract"
    note: str    # caveats, e.g. "claim number inferred"
    text: str
    score: float


@dataclass
class Result:
    rank: int
    patent_id: str
    title: str
    classification: str
    publication_date: str
    url: str
    score: float
    excerpt_label: str   # the best-matching piece: "Claim 4", "Description para 12"
    excerpt_note: str    # caveats about it, e.g. "claim number inferred"
    excerpt_text: str
    # Every piece shown, best first (empty when there was no query).
    excerpts: list[Excerpt] = field(default_factory=list)
    # Further pieces within EXCERPT_MARGIN of the best that were not shown.
    more_close_matches: int = 0


@dataclass
class Suggestion:
    """A way out of an empty result: what to change, and what it would give."""

    kind: str          # spelling | similar_title | classification | match_any |
                       # search_by_meaning | drop_filter
    message: str
    query: str         # the query to run instead
    filters: Filters   # the filters to run instead
    patents: int       # how many patents would pass the filters


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
    # True when the filters matched only patents excluded from the results
    # (in a "similar to this patent" search: only the source patent itself).
    excluded_only: bool = False
    suggestions: list[Suggestion] = field(default_factory=list)


class PatentSearch:
    def __init__(self, index_dir: str = "index", model=None):
        self.hnsw_path = os.path.join(index_dir, HNSW_FILE)
        # Checked once: a file-system check on every search cost ~0.1 ms on Windows.
        self._has_hnsw = os.path.exists(self.hnsw_path)
        self._hnsw = None
        self._date_order: list[tuple[str, str]] | None = None   # (date, patent id), sorted
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

        # Raw (unstemmed) words per field, for spelling suggestions. Built from
        # the stored titles and abstracts, so every word is in the indexes.
        def words(texts) -> set[str]:
            return {w for text in texts for w in TOKEN_RE.findall(text.lower())
                    if len(w) > 2 and not w.isdigit()}
        self.title_vocabulary = words(p["title"] for p in self.patents.values())
        self.abstract_vocabulary = words(p["abstract"] for p in self.patents.values())
        self.titles_by_key = {}
        for patent in self.patents.values():
            self.titles_by_key.setdefault(normalize_title(patent["title"]), patent["title"])

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
            from .model import revision_for
            self._model = SentenceTransformer(self.model_name,
                                              revision=revision_for(self.model_name))
        return self._model

    @property
    def has_hnsw(self) -> bool:
        return self._has_hnsw

    @property
    def hnsw(self):
        """The HNSW graph, loaded on first use (FAISS is only imported then)."""
        if self._hnsw is None:
            import faiss
            graph = faiss.read_index(self.hnsw_path)
            if graph.ntotal != len(self.vectors):
                raise ValueError(f"{self.hnsw_path} holds {graph.ntotal} vectors but the index "
                                 f"has {len(self.vectors)}; rebuild the index")
            self._hnsw = graph
        return self._hnsw

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True,
                                 normalize_embeddings=True).astype(np.float32)[0]

    # -- "similar to this patent" -----------------------------------------
    def resolve_patent_id(self, raw: str) -> tuple[str | None, list[str]]:
        """Match a typed patent ID to the corpus; (id, []) or (None, close IDs).

        IDs are stored as bare publication numbers ("20240051333"). Users
        copy them in many shapes: "US20240051333A1", "US 2024/0051333 A1",
        "us2024-0051333". Case, spaces and punctuation are ignored, and a
        leading country code and trailing kind code are stripped.
        """
        key = re.sub(r"[^A-Z0-9]", "", raw.upper())
        key = re.sub(r"^US", "", key)
        key = re.sub(r"(?<=\d)[ABCPSE]\d?$", "", key)
        if key in self.patents:
            return key, []
        return None, difflib.get_close_matches(key, list(self.patents), n=3, cutoff=0.8)

    def patent_vector(self, patent_id: str) -> tuple[np.ndarray, list[str]]:
        """A query vector describing a patent, and the parts it was built from.

        The abstract says what the invention is in plain words; the first
        claim says what is actually protected. Each part is averaged over
        its pieces (a long text is stored as several), and the two parts are
        weighted equally. A patent with neither falls back to its
        description. Built from stored vectors, so no model is needed.
        """
        start, end = (self.unit_rows_of(patent_id)
                      if patent_id in self.patent_position else (0, 0))
        if start == end:
            raise ValueError(f"patent {patent_id} has no indexed text")
        rows = range(start, end)

        def unit_mean(selected: list[int]) -> np.ndarray:
            vector = self.vectors[selected].mean(axis=0)
            return vector / max(float(np.linalg.norm(vector)), 1e-12)

        parts, used = [], []
        abstract = [r for r in rows if self.units[r]["kind"] == "abstract"]
        if abstract:
            parts.append(unit_mean(abstract))
            used.append("Abstract")
        claims = [r for r in rows if self.units[r]["kind"] == "claim"]
        if claims:
            first = min(self.units[r]["position"] for r in claims)
            parts.append(unit_mean([r for r in claims if self.units[r]["position"] == first]))
            used.append(f"Claim {first}")
        if not parts:
            parts.append(unit_mean(list(rows)))
            used.append("Description")
        vector = np.mean(parts, axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return vector.astype(np.float32), used

    def search_similar(self, patent_id: str, filters: Filters | None = None,
                       top_k: int = 10, strategy: str = "auto",
                       max_excerpts: int = MAX_EXCERPTS) -> tuple[SearchReport, list[str]]:
        """Patents most similar to patent_id, never including itself.

        Patents with an identical abstract are kept, not hidden: identical
        text doesn't prove they are the same patent family (DECISIONS.md D13).
        """
        vector, used = self.patent_vector(patent_id)
        report = self.search(filters=filters, top_k=top_k, strategy=strategy,
                             query_vector=vector, max_excerpts=max_excerpts,
                             exclude={patent_id})
        return report, used

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

    # Each keyword field: (Filters attribute, label, inverted index, phrase sources).
    def _keyword_fields(self):
        return [
            ("title_keywords", "title", self.title_index, [self.title_phrases]),
            ("abstract_keywords", "abstract", self.abstract_index, [self.abstract_phrases]),
            ("title_or_abstract_keywords", "title/abstract", self.title_or_abstract_index,
             [self.title_phrases, self.abstract_phrases]),
        ]

    def _filter_hits(self, filters: Filters) -> list[tuple[str, str, set[str]]]:
        """(Filters attribute, display name, matching patents) per active filter."""
        found: list[tuple[str, str, set[str]]] = []
        if filters.classification:
            codes = self._classification_codes(filters.classification)
            hits: set[str] = set()
            for code in codes:  # several codes are OR-ed together
                hits |= self.classification_prefix.get(code, set())
            found.append(("classification", f"classification {'/'.join(codes)}", hits))
        if filters.exact_title:
            key = normalize_title(filters.exact_title)
            found.append(("exact_title", f'title is "{filters.exact_title}"',
                          set(self.exact_title_index.get(key, ()))))
        for attribute, label, inverted, phrases in self._keyword_fields():
            raw = getattr(filters, attribute)
            if raw:
                found.append((attribute, f'{label} "{raw}"', self._keyword_match(
                    parse_keywords(raw), inverted, phrases, filters.match_any)))
        if filters.published_from or filters.published_to:
            low = date_bound(filters.published_from, end=False) if filters.published_from else ""
            high = date_bound(filters.published_to, end=True) if filters.published_to else "9999"
            # Patents sorted by date once; a range is two binary searches.
            dates = self._dates_sorted
            first = bisect.bisect_left(dates, (low, ""))
            last = bisect.bisect_right(dates, (high, "\uffff"))
            hits = {pid for date, pid in dates[first:last] if date}
            span = f"{filters.published_from or '…'} to {filters.published_to or '…'}"
            found.append(("published", f"published {span}", hits))
        return found

    @property
    def _dates_sorted(self) -> list[tuple[str, str]]:
        if self._date_order is None:
            self._date_order = sorted((p["publication_date"] or "", pid)
                                      for pid, p in self.patents.items())
        return self._date_order

    def resolve_filters(self, filters: Filters) -> tuple[set[str] | None, dict[str, int], str]:
        """Return (allowed patent ids, per-filter counts, name of an empty filter).

        None means "no filter applied", which is different from an empty set
        ("filters applied and nothing matched").
        """
        counts: dict[str, int] = {}
        allowed: set[str] | None = None
        empty_reason = ""
        for _, name, hits in self._filter_hits(filters):
            counts[name] = len(hits)
            if not hits and not empty_reason:
                empty_reason = name
            allowed = hits if allowed is None else allowed & hits
        if allowed is not None:
            counts["combined"] = len(allowed)
        return allowed, counts, empty_reason

    # -- suggestions when the filters leave nothing -----------------------
    def _count(self, filters: Filters) -> int:
        allowed, _, _ = self.resolve_filters(filters)
        return len(self.patents) if allowed is None else len(allowed)

    def suggest(self, query: str, filters: Filters, limit: int = 6) -> list[Suggestion]:
        """Ways out of an empty result, each with the patents it would give.

        Only changes that would actually return patents are suggested. In
        order: likely typos, near titles, nearby classification codes,
        "any word" instead of "all words", searching the words by meaning,
        and dropping one filter.
        """
        found = self._filter_hits(filters)
        suggestions: list[Suggestion] = []

        def offer(kind: str, message: str, new_filters: Filters, new_query: str = query) -> None:
            count = self._count(new_filters)
            if count and not any(s.filters == new_filters and s.query == new_query
                                 for s in suggestions):
                suggestions.append(Suggestion(kind, message.replace("{count}", str(count)),
                                              new_query, new_filters, count))

        # 1. Spelling: a keyword that appears nowhere in its field.
        for attribute, label, inverted, _ in self._keyword_fields():
            raw = getattr(filters, attribute)
            vocabulary = (self.title_vocabulary if label == "title" else
                          self.abstract_vocabulary if label == "abstract" else
                          self.title_vocabulary | self.abstract_vocabulary)
            for display, tokens in parse_keywords(raw):
                if len(tokens) != 1 or inverted.get(tokens[0]):
                    continue
                for candidate in difflib.get_close_matches(display.lower(), vocabulary,
                                                           n=3, cutoff=0.8):
                    if stem(candidate) == tokens[0]:
                        continue
                    fixed = re.sub(rf"(?<![\w\"]){re.escape(display)}(?![\w\"])",
                                   candidate, raw, count=1)
                    offer("spelling", f'Did you mean "{candidate}" in the {label}? '
                          "→ {count} patents", replace(filters, **{attribute: fixed}))

        for attribute, name, hits in found:
            if hits:
                continue
            # 2. A title that is close to the one typed.
            if attribute == "exact_title":
                for key in difflib.get_close_matches(normalize_title(filters.exact_title),
                                                     self.titles_by_key, n=3, cutoff=0.75):
                    title = self.titles_by_key[key]
                    offer("similar_title", f'Did you mean the title "{title}"? → {{count}} patents',
                          replace(filters, exact_title=title))
            # 3. Classification codes that do exist next to the one typed.
            if attribute == "classification":
                for code in self._classification_codes(filters.classification):
                    parent = code
                    while parent and parent not in self.classification_prefix:
                        parent = parent[:-1]
                    if not parent:
                        continue
                    children = {}
                    for prefix, ids in self.classification_prefix.items():
                        if len(prefix) == min(len(code), len(parent) + 1) and \
                                prefix.startswith(parent) and prefix != code:
                            children[prefix] = len(ids)
                    for child in sorted(children, key=lambda c: -children[c])[:3]:
                        offer("classification", f"Classification {child} exists → "
                              "{count} patents", replace(filters, classification=child))

        # 4. "Any word" instead of "all words", when only some words match.
        if not filters.match_any:
            for attribute, label, inverted, phrases in self._keyword_fields():
                terms = parse_keywords(getattr(filters, attribute))
                if len(terms) < 2:
                    continue
                per_term = [(display, len(self._keyword_match([(display, tokens)], inverted,
                                                             phrases, False)))
                            for display, tokens in terms]
                # Each word may match some patents while no patent has them all.
                all_words = self._keyword_match(terms, inverted, phrases, False)
                if not all_words and any(n for _, n in per_term):
                    detail = ", ".join(f'"{d}" {n}' for d, n in per_term)
                    offer("match_any", f"In the {label}, {detail}: match any word instead "
                          "→ {count} patents", replace(filters, match_any=True))
                    break

        # 5 and 6. Per empty filter: search its words by meaning, or drop it.
        # When every filter matches something but no patent passes them all,
        # each filter is a candidate to drop.
        none_empty = all(hits for _, _, hits in found)
        for attribute, name, hits in found:
            if hits and not none_empty:
                continue
            if attribute == "published":
                without = replace(filters, published_from="", published_to="")
            else:
                without = replace(filters, **{attribute: ""})
            if attribute not in ("classification", "published") and not hits:
                words = getattr(filters, attribute).replace('"', "")
                new_query = f"{query} {words}".strip()
                offer("search_by_meaning", f'Search "{words}" by meaning instead of requiring '
                      "the exact words (ranks all text of {count} patents)",
                      without, new_query)
            if attribute == "published":
                dated = [date for date, _ in self._dates_sorted if date]
                coverage = (f" (this index covers {dated[0]} to {dated[-1]})" if dated
                            else " (this index has no publication dates)")
                offer("drop_filter", f"Remove the {name} filter{coverage} → {{count}} patents",
                      without)
                continue
            offer("drop_filter", f"Remove the {name} filter → {{count}} patents", without)

        return suggestions[:limit]

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
        date = patent["publication_date"] or ""
        if filters.published_from and not (date and date >= date_bound(filters.published_from, False)):
            return False
        if filters.published_to and not (date and date <= date_bound(filters.published_to, True)):
            return False
        return True

    # -- ranking ----------------------------------------------------------
    @staticmethod
    def _top_k(best: np.ndarray, top_k: int) -> np.ndarray:
        """Positions of the top_k highest values, best first, ties by position.

        Ties are real in this data: identical passages appear in several
        patents and get identical scores. argpartition alone would pick an
        arbitrary subset of the tied patents at the cut-off, so everything
        tied with the k-th score is kept and the tie is broken by position.
        """
        usable = np.flatnonzero(np.isfinite(best))
        if usable.size > top_k:
            kth = np.partition(best[usable], usable.size - top_k)[usable.size - top_k]
            usable = usable[best[usable] >= kth]
        return usable[np.lexsort((usable, -best[usable]))][:top_k]

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

    def _rank_hnsw(self, query_vector: np.ndarray, allowed_mask: np.ndarray | None,
                   top_k: int) -> tuple[list[tuple[str, float, int]], int]:
        """Best passage per patent from the HNSW graph, filters applied inside.

        One patent can own many of the returned passages, so the search is
        repeated twice as wide until top_k distinct patents come back. With a
        filter, excluded passages are skipped while the graph is walked, so
        the search is widened in proportion to the share of passages allowed.

        The graph only chooses the candidate patents. The best 3 x top_k are
        then rescored exactly over all their passages, so its score, its best passage
        and the evidence shown are the same as exact search gives; without
        this, a patent found through a nearly-best passage showed that passage.
        """
        import faiss

        keep, selector, row_allowed = None, None, None
        available = len(self.vectors)
        if allowed_mask is not None:
            row_allowed = allowed_mask[self.unit_patent]
            available = int(row_allowed.sum())
            if not available:
                return [], 0
            keep = np.packbits(row_allowed, bitorder="little")
            selector = faiss.IDSelectorBitmap(len(self.vectors), faiss.swig_ptr(keep))
        k = int(np.ceil(HNSW_CANDIDATES / max(available / len(self.vectors), 0.01)))
        while True:
            width = min(k, available)
            params = faiss.SearchParametersHNSW()
            params.efSearch = width
            if selector is not None:
                params.sel = selector
            scores, rows = self.hnsw.search(query_vector[None, :].astype(np.float32),
                                            width, params=params)
            best: dict[str, tuple[float, int]] = {}
            for score, row in zip(scores[0], rows[0]):
                if row < 0 or (row_allowed is not None and not row_allowed[row]):
                    continue   # the second check: a filter can never be loosened
                patent = self.patent_order[self.unit_patent[row]]
                best.setdefault(patent, (float(score), int(row)))   # best first
            if len(best) >= top_k or width >= available:
                break
            k *= 2
        del keep
        rescored = []
        for patent_id in list(best)[:top_k * HNSW_RESCORE_FACTOR]:   # best first
            start, end = self.unit_rows_of(patent_id)
            scores = self.vectors[start:end] @ query_vector
            top = int(np.argmax(scores))      # first maximum, as exact search breaks ties
            rescored.append((patent_id, float(scores[top]), start + top))
        rescored.sort(key=lambda item: (-item[1], self.patent_position[item[0]]))
        return rescored[:top_k], width

    def _rank_postfilter(self, scores: np.ndarray, filters: Filters, top_k: int,
                         exclude: frozenset[str] = frozenset()
                         ) -> list[tuple[str, float, int]]:
        """Naive baseline: rank every unit, then filter while walking down."""
        decided: dict[str, bool] = {}
        ranked: list[tuple[str, float, int]] = []
        for row in np.argsort(-scores, kind="stable"):
            patent_id = self.patent_order[self.unit_patent[row]]
            if patent_id not in decided:
                decided[patent_id] = (patent_id not in exclude
                                      and self._passes_without_index(patent_id, filters))
                if decided[patent_id]:
                    ranked.append((patent_id, float(scores[row]), int(row)))
                    if len(ranked) >= top_k:
                        break
        return ranked

    # -- search -----------------------------------------------------------
    def search(self, query: str = "", filters: Filters | None = None, top_k: int = 10,
               strategy: str = "auto",
               query_vector: np.ndarray | None = None,
               max_excerpts: int = MAX_EXCERPTS,
               exclude: set[str] | frozenset[str] = frozenset()) -> SearchReport:
        """Run one search.

        query_vector lets a caller that already embedded the query (the
        benchmark, or a "similar to this patent" search) skip the embedding
        step. exclude lists patents that must never be returned, such as the
        source patent of a similarity search. max_excerpts caps how many
        matching pieces each result shows.
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
            if exclude:
                matched_something = allowed is None or bool(allowed)
                allowed = (set(self.patents) if allowed is None else allowed) - set(exclude)
                if not allowed and matched_something:
                    # The filters did match, but only excluded patents: no
                    # filter change would help, so no suggestions either.
                    report.excluded_only = True
                    timings["total_ms"] = (time.perf_counter() - total_start) * 1000
                    return report
            timings["filter_ms"] = (time.perf_counter() - start) * 1000
            if allowed is not None and not allowed:
                report.empty_reason = empty_reason
                start = time.perf_counter()
                report.suggestions = self.suggest(query, filters)
                timings["suggest_ms"] = (time.perf_counter() - start) * 1000
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

        if strategy == "hnsw" and not self.has_hnsw:
            raise ValueError("this index has no HNSW graph; rebuild it with --hnsw")
        exact_only = strategy == "exact"
        if exact_only:
            strategy = "auto"
        if strategy == "auto" and not naive and not exact_only and self.has_hnsw \
                and not (filters and filters.active()):
            strategy = "hnsw"      # no filter: only an exclusion (similar-patent search)
        if strategy == "auto":
            share = report.candidate_units / max(len(self.units), 1)
            strategy = ("prefilter" if allowed is not None
                        and share <= AUTO_PREFILTER_MAX_SHARE else "mask")

        if naive:
            start = time.perf_counter()
            scores = self.vectors @ query_vector
            timings["score_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            ranked = self._rank_postfilter(scores, filters, top_k, frozenset(exclude))
            timings["rank_ms"] = (time.perf_counter() - start) * 1000
            report.scored_units = len(self.units)
            report.candidate_units = len(self.units)
            report.candidate_patents = len(self.patents) - len(exclude)
            report.strategy = "postfilter"
        elif strategy == "hnsw":
            start = time.perf_counter()
            ranked, report.scored_units = self._rank_hnsw(query_vector, allowed_mask, top_k)
            # Graph search and grouping are interleaved, so reported together.
            timings["score_ms"] = (time.perf_counter() - start) * 1000
            timings["rank_ms"] = 0.0
            report.strategy = "hnsw"
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
        report.results = [self._as_result(rank, patent_id, score, row, query_vector, max_excerpts)
                          for rank, (patent_id, score, row) in enumerate(ranked, start=1)]
        timings["evidence_ms"] = (time.perf_counter() - start) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        return report

    def _unit_excerpt(self, row: int, score: float) -> Excerpt:
        unit = self.units[row]
        notes = [unit["label_note"]] if unit.get("label_note") else []
        if unit.get("context_label"):
            notes.append(f"with {unit['context_label']} as context")
        return Excerpt(label=unit["label"], note="; ".join(notes),
                       text=unit["display_text"], score=round(float(score), 4))

    def _excerpts(self, patent_id: str, best_row: int, query_vector: np.ndarray,
                  max_excerpts: int) -> tuple[list[Excerpt], int]:
        """The patent's best piece, then other pieces scoring close to it.

        Scores only this patent's own block of rows (about 54), so it costs
        the same whichever filtering strategy ranked the patent. Pieces that
        are another part of a claim or paragraph already shown, or repeat its
        text, are skipped: they would show the same evidence twice.
        """
        start, end = self.unit_rows_of(patent_id)
        scores = self.vectors[start:end] @ query_vector
        best_score = float(scores[best_row - start])
        best_unit = self.units[best_row]
        shown = [self._unit_excerpt(best_row, best_score)]
        seen_keys = {(best_unit["kind"], best_unit["position"])}
        seen_texts = {best_unit["display_text"]}
        more = 0
        for offset in np.argsort(-scores, kind="stable"):
            row = start + int(offset)
            if row == best_row:
                continue
            if best_score - float(scores[offset]) > EXCERPT_MARGIN:
                break  # sorted, so nothing further down is close enough
            unit = self.units[row]
            key = (unit["kind"], unit["position"])
            if key in seen_keys or unit["display_text"] in seen_texts:
                continue
            seen_keys.add(key)
            seen_texts.add(unit["display_text"])
            if len(shown) < max_excerpts:
                shown.append(self._unit_excerpt(row, scores[offset]))
            else:
                more += 1
        return shown, more

    def unit_rows_of(self, patent_id: str) -> tuple[int, int]:
        position = self.patent_position[patent_id]
        return int(self.block_starts[position]), int(self.block_ends[position])

    def _as_result(self, rank: int, patent_id: str, score: float, row: int | None,
                   query_vector: np.ndarray | None = None,
                   max_excerpts: int = MAX_EXCERPTS) -> Result:
        patent = self.patents[patent_id]
        excerpts: list[Excerpt] = []
        more = 0
        if row is None:
            label, note, text = "Abstract", "", patent["abstract"]
        else:
            best = self._unit_excerpt(row, score)
            label, note, text = best.label, best.note, best.text
            if query_vector is not None:
                excerpts, more = self._excerpts(patent_id, row, query_vector, max(1, max_excerpts))
        return Result(rank=rank, patent_id=patent_id, title=patent["title"],
                      classification=patent["classification"],
                      publication_date=patent["publication_date"], url=patent["url"],
                      score=round(score, 4), excerpt_label=label,
                      excerpt_note=note, excerpt_text=text,
                      excerpts=excerpts, more_close_matches=more)
