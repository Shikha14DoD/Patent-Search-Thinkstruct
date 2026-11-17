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


# -- filter edge cases ----------------------------------------------------------
@pytest.fixture
def titles(tmp_path):
    """Five patents whose titles exercise phrase and exact-title boundaries."""
    def patent(pid, title, code):
        return make_patent(pid, title, f"Abstract of {title.lower()}.", code, ["1 . A part."])
    patents = [patent("1", "WHEEL HUB ASSEMBLY", "B60B2700"),
               patent("2", "FLYWHEEL HUB", "F16F1500"),
               patent("3", "HUB FOR A WHEEL", "B60B2702"),
               patent("4", "WHEEL HUB", "B60C2304"),
               patent("5", "A WHEEL-HUB UNIT", "B60D1000")]
    return make_engine(tmp_path, patents, {})


def allowed(engine, **kwargs):
    ids, _, _ = engine.resolve_filters(Filters(**kwargs))
    return sorted(ids)


def test_phrase_matches_whole_words_at_start_middle_end_and_whole_field(titles):
    # start ("WHEEL HUB ASSEMBLY"), whole field ("WHEEL HUB"), middle ("A WHEEL-HUB UNIT");
    # "FLYWHEEL HUB" must not match: "wheel" is only part of a word there.
    assert allowed(titles, title_keywords='"wheel hub"') == ["1", "4", "5"]
    assert allowed(titles, title_keywords='"hub assembly"') == ["1"]   # end of the field
    assert allowed(titles, title_keywords='"hub wheel"') == []          # order matters


def test_words_without_quotes_need_not_be_adjacent(titles):
    assert allowed(titles, title_keywords="hub wheel") == ["1", "3", "4", "5"]


def test_match_any_versus_all(titles):
    assert allowed(titles, title_keywords="assembly unit") == []
    assert allowed(titles, title_keywords="assembly unit", match_any=True) == ["1", "5"]


def test_several_classification_prefixes_are_or_ed(titles):
    assert allowed(titles, classification="B60B, F16F") == ["1", "2", "3"]
    assert allowed(titles, classification="b60") == ["1", "3", "4", "5"]   # case-insensitive


def test_filters_combine_with_and(titles):
    assert allowed(titles, classification="B60B", title_keywords="wheel") == ["1", "3"]
    assert allowed(titles, classification="B60B", title_keywords='"wheel hub"') == ["1"]


def test_exact_title_ignores_case_and_punctuation_only(titles):
    assert allowed(titles, exact_title="wheel hub") == ["4"]
    assert allowed(titles, exact_title="Wheel-Hub") == ["4"]
    assert allowed(titles, exact_title="wheel hubs") == []   # no stemming in exact titles


def test_unknown_classification_matches_nothing(titles):
    assert allowed(titles, classification="H04L") == []


def test_filter_only_search_never_loads_the_model(titles):
    titles.search(filters=Filters(classification="B60B"))
    titles.search(filters=Filters(exact_title="WHEEL HUB"))
    assert titles._model is None


def test_same_search_twice_gives_identical_results(engine):
    first = engine.search(query_vector=QUERY, top_k=2)
    second = engine.search(query_vector=QUERY, top_k=2)
    assert [(r.patent_id, [e.label for e in r.excerpts]) for r in first.results] == \
           [(r.patent_id, [e.label for e in r.excerpts]) for r in second.results]


# -- "similar to this patent" ---------------------------------------------------
@pytest.fixture
def family(tmp_path):
    """Real-looking IDs; 2 is the closest to 1, 3 is in another class, 4 has no abstract."""
    patents = [
        make_patent("20240000001", "SPOKE", "A hollow spoke.", "B60B2700", ["1 . A spoke."]),
        make_patent("20240000002", "SPOKED WHEEL", "A wheel with spokes.", "B60B2700",
                    ["1 . A wheel with a spoke."]),
        make_patent("20240000003", "TIRE", "A tire.", "B60C2304", ["1 . A tire."]),
        make_patent("20240000004", "HUB", "", "B60B2702", ["1 . A hub."]),
    ]
    scores = {("20240000001", "Abstract"): 0.9, ("20240000001", "Claim 1"): 0.9,
              ("20240000002", "Abstract"): 0.85, ("20240000002", "Claim 1"): 0.8,
              ("20240000003", "Abstract"): 0.1, ("20240000003", "Claim 1"): 0.1,
              ("20240000004", "Claim 1"): 0.5}
    return make_engine(tmp_path, patents, scores)


def test_similar_search_never_returns_the_source_patent(family):
    report, used = family.search_similar("20240000001", top_k=10)
    ids = [r.patent_id for r in report.results]
    assert "20240000001" not in ids and ids[0] == "20240000002" and len(ids) == 3
    assert used == ["Abstract", "Claim 1"]
    assert family._model is None                  # built from stored vectors


def test_patent_ids_are_accepted_in_common_shapes(family):
    for typed in ("20240000001", "US20240000001A1", "us 2024/0000001 a1", "US2024-0000001"):
        assert family.resolve_patent_id(typed) == ("20240000001", [])
    found, close = family.resolve_patent_id("2024000000")   # a digit short
    assert found is None and close


def test_similar_search_respects_filters_and_every_strategy_agrees(family):
    outcomes = {
        strategy: [r.patent_id for r in family.search_similar(
            "20240000001", Filters(classification="B60B"), strategy=strategy)[0].results]
        for strategy in ("prefilter", "mask", "postfilter", "auto")
    }
    assert outcomes["mask"] == ["20240000002", "20240000004"]
    assert len(set(map(tuple, outcomes.values()))) == 1


def test_filters_matching_only_the_source_say_so(family):
    report, _ = family.search_similar("20240000001", Filters(exact_title="SPOKE"))
    assert report.results == [] and report.excluded_only and report.suggestions == []


def test_patent_without_abstract_uses_its_claim(family):
    _, used = family.search_similar("20240000004")
    assert used == ["Claim 1"]


# -- publication date filter and newest-first ordering --------------------

def test_date_bounds_accept_year_month_and_day():
    from src.search import date_bound
    assert date_bound("2024", end=False) == "2024-01-01"
    assert date_bound("2024", end=True) == "2024-12-31"
    assert date_bound("2024-03", end=True) == "2024-03-31"
    assert date_bound("2024-03-15", end=False) == "2024-03-15"
    with pytest.raises(ValueError):
        date_bound("24-03", end=False)


@pytest.fixture
def dated_engine(tmp_path):
    """Three patents published in different months, plus one with no date."""
    patents = []
    for patent_id, date in (("1", "2024-02-15"), ("2", "2024-06-20"), ("3", "2025-01-09"),
                            ("4", "")):
        patent = make_patent(patent_id, f"WHEEL {patent_id}", "A wheel.", "B60B2700",
                             ["1 . A wheel comprising a rim."])
        patent.publication_date = date
        patents.append(patent)
    scores = {("1", "Claim 1"): 0.9, ("2", "Claim 1"): 0.8, ("3", "Claim 1"): 0.7,
              ("4", "Claim 1"): 0.6}
    return make_engine(tmp_path, patents, scores)


def test_date_range_keeps_only_patents_published_inside_it(dated_engine):
    report = dated_engine.search(query_vector=QUERY, filters=Filters(published_from="2024-03",
                                                                     published_to="2024"))
    assert [r.patent_id for r in report.results] == ["2"]
    year = dated_engine.search(query_vector=QUERY, filters=Filters(published_from="2025"))
    assert [r.patent_id for r in year.results] == ["3"]      # the undated patent never passes


def test_an_empty_date_range_says_what_the_index_covers(dated_engine):
    report = dated_engine.search(query_vector=QUERY, filters=Filters(published_to="2023"))
    assert not report.results and report.empty_reason.startswith("published")
    drop = [s for s in report.suggestions if s.kind == "drop_filter"][0]
    assert "2024-02-15 to 2025-01-09" in drop.message
    assert drop.filters.published_from == "" and drop.filters.published_to == ""


def test_date_filter_agrees_with_the_naive_method(dated_engine):
    filters = Filters(published_from="2024-06-01", published_to="2025-12")
    exact = dated_engine.search(query_vector=QUERY, filters=filters, strategy="prefilter")
    naive = dated_engine.search(query_vector=QUERY, filters=filters, strategy="postfilter")
    assert [r.patent_id for r in exact.results] == [r.patent_id for r in naive.results] == ["2", "3"]


def test_newest_first_reorders_the_same_results(dated_engine):
    from src.search import newest_first
    report = dated_engine.search(query_vector=QUERY, top_k=3)
    assert [r.patent_id for r in report.results] == ["1", "2", "3"]       # by score
    ordered = newest_first(report.results)
    assert [r.patent_id for r in ordered] == ["3", "2", "1"]
    assert [r.rank for r in ordered] == [1, 2, 3]
