"""Tests for the search engine, on a tiny hand-made index.

Vectors are written by hand so every score is known in advance: a unit
meant to score s against the query [1, 0, 0] gets the vector
[s, sqrt(1 - s^2), 0]. No model is loaded.

Run with:  python -m pytest tests -q
"""

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_index import assemble_bundle  # noqa: E402
from src.load_data import Patent, TokenCounter, build_units, parse_claims  # noqa: E402
from src.search import Filters, PatentSearch  # noqa: E402

QUERY = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def make_patent(patent_id, title, abstract, classification, claims):
    return Patent(patent_id=patent_id, title=title, abstract=abstract,
                  classification=classification, publication_date="2024-01-01",
                  url=f"https://example.com/{patent_id}", claims=parse_claims(claims))


def make_engine(tmp_path, patents, scores):
    """Build an index where unit (patent_id, label) scores scores[...] against QUERY."""
    units, _ = build_units(patents, TokenCounter())
    vectors = np.zeros((len(units), 3), dtype=np.float32)
    for row, unit in enumerate(units):
        s = scores.get((unit.patent_id, unit.label), 0.0)
        vectors[row] = [s, math.sqrt(1 - s * s), 0.0]
    with open(tmp_path / "index.pkl", "wb") as handle:
        pickle.dump(assemble_bundle(patents, units, "test-model"), handle)
    np.save(tmp_path / "vectors.npy", vectors)
    return PatentSearch(str(tmp_path))


@pytest.fixture
def engine(tmp_path):
    wheel = make_patent(
        "1", "WHEEL WITH SPOKES", "A wheel with hollow spokes.", "B60B2700",
        ["1 . A wheel comprising " + "word " * 300,  # long: split into two parts
         "2 . The wheel of claim 1 , wherein the spoke is hollow.",
         "3 . The wheel of claim 1 , wherein the spoke is steel.",
         "4 . The wheel of claim 1 , wherein the rim is steel.",
         "5 . The wheel of claim 1 , wherein the hub is cast."])
    tire = make_patent(
        "2", "PNEUMATIC TIRE", "A tire with a pressure sensor.", "B60C2304",
        ["1 . A tire comprising a tread and a pressure sensor."])
    scores = {
        ("1", "Abstract"): 0.30,
        ("1", "Claim 1 (part 1)"): 0.80,
        ("1", "Claim 1 (part 2)"): 0.79,   # same claim: must not be shown twice
        ("1", "Claim 2"): 0.76,
        ("1", "Claim 3"): 0.74,
        ("1", "Claim 4"): 0.73,            # close, but over the 3-excerpt cap
        ("1", "Claim 5"): 0.60,            # more than 0.10 below the best
        ("2", "Abstract"): 0.50,
        ("2", "Claim 1"): 0.55,
    }
    return make_engine(tmp_path, [wheel, tire], scores)


# -- several excerpts per result ---------------------------------------------
def test_results_show_the_best_piece_then_close_ones(engine):
    report = engine.search(query_vector=QUERY, top_k=2)
    first, second = report.results
    assert first.patent_id == "1" and second.patent_id == "2"
    assert [e.label for e in first.excerpts] == ["Claim 1 (part 1)", "Claim 2", "Claim 3"]
    assert first.excerpts[0].score == pytest.approx(0.80, abs=1e-4)
    assert first.more_close_matches == 1          # Claim 4; Claim 5 is too far below
    assert [e.label for e in second.excerpts] == ["Claim 1", "Abstract"]


def test_excerpt_cap_is_respected(engine):
    report = engine.search(query_vector=QUERY, top_k=1, max_excerpts=1)
    (result,) = report.results
    assert [e.label for e in result.excerpts] == ["Claim 1 (part 1)"]
    assert result.more_close_matches == 3         # Claims 2, 3 and 4


def test_every_strategy_gives_the_same_excerpts(engine):
    filters = Filters(classification="B60")
    outcomes = {
        strategy: [(r.patent_id, [e.label for e in r.excerpts])
                   for r in engine.search(filters=filters, query_vector=QUERY,
                                          top_k=2, strategy=strategy).results]
        for strategy in ("prefilter", "mask", "postfilter")
    }
    assert outcomes["prefilter"] == outcomes["mask"] == outcomes["postfilter"]


def test_filter_only_search_shows_an_abstract_preview(engine):
    report = engine.search(filters=Filters(classification="B60C"), top_k=5)
    (result,) = report.results
    assert result.excerpts == [] and result.excerpt_label == "Abstract"


# -- suggestions when filters match nothing -----------------------------------
def kinds(report):
    return [(s.kind, s.patents) for s in report.suggestions]


def test_misspelled_keyword_gets_a_spelling_suggestion(engine):
    report = engine.search(filters=Filters(title_keywords="spokse"))
    assert report.results == [] and report.empty_reason
    first = report.suggestions[0]
    assert first.kind == "spelling" and first.filters.title_keywords == "spokes"
    assert first.patents == 1


def test_words_that_match_separately_suggest_any_word(engine):
    report = engine.search(filters=Filters(abstract_keywords="sensor hollow"))
    any_word = [s for s in report.suggestions if s.kind == "match_any"]
    assert any_word and any_word[0].filters.match_any and any_word[0].patents == 2
    assert '"sensor" 1' in any_word[0].message and '"hollow" 1' in any_word[0].message


def test_unknown_classification_suggests_codes_that_exist(engine):
    report = engine.search(filters=Filters(classification="B60X"))
    offered = {s.filters.classification for s in report.suggestions if s.kind == "classification"}
    assert offered == {"B60B", "B60C"}


def test_near_title_is_suggested(engine):
    report = engine.search(filters=Filters(exact_title="pneumatic tyre"))
    titles = [s.filters.exact_title for s in report.suggestions if s.kind == "similar_title"]
    assert titles[0] == "PNEUMATIC TIRE"


def test_filter_words_can_move_into_the_query(engine):
    report = engine.search("wheel", filters=Filters(title_keywords="aerodynamic"))
    by_meaning = [s for s in report.suggestions if s.kind == "search_by_meaning"]
    assert by_meaning[0].query == "wheel aerodynamic"
    assert by_meaning[0].filters.title_keywords == ""


def test_filters_that_never_overlap_suggest_dropping_each(engine):
    report = engine.search(filters=Filters(classification="B60C", title_keywords="wheel"))
    assert report.empty_reason == ""            # each filter matches something
    assert sorted(kinds(report)) == [("drop_filter", 1), ("drop_filter", 1)]


def test_no_suggestions_when_there_are_results(engine):
    report = engine.search(filters=Filters(classification="B60B"))
    assert report.results and report.suggestions == []