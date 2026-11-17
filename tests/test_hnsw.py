"""Tests for the HNSW (approximate) search strategy.

Run with:  python -m pytest tests -q
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("faiss")

from src.build_index import assemble_bundle, build_hnsw  # noqa: E402
from src.load_data import TokenCounter, build_units  # noqa: E402
from src.search import HNSW_FILE, Filters, PatentSearch  # noqa: E402
from tests.test_search import make_patent  # noqa: E402


def build(tmp_path, patents, seed=0, graph=True):
    units, _ = build_units(patents, TokenCounter())
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(units), 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with open(tmp_path / "index.pkl", "wb") as handle:
        pickle.dump(assemble_bundle(patents, units, "test-model"), handle)
    np.save(tmp_path / "vectors.npy", vectors)
    if graph:
        build_hnsw(vectors, str(tmp_path / HNSW_FILE))
    return PatentSearch(str(tmp_path))


@pytest.fixture
def engine(tmp_path):
    # 40 patents: 30 in B60B (75%), 10 in A61B (25%).
    patents = [make_patent(str(i), f"WHEEL {i}", f"A wheel number {i}.",
                           "B60B100FI" if i < 30 else "A61B5/097",
                           [f"1 . A wheel comprising part {i}.", "2 . The wheel of claim 1 ."])
               for i in range(40)]
    return build(tmp_path, patents)


def query(seed):
    q = np.random.default_rng(seed).normal(size=16).astype(np.float32)
    return q / np.linalg.norm(q)


def ids(report):
    return [r.patent_id for r in report.results]


def test_auto_uses_hnsw_without_filters_and_matches_exact(engine):
    for seed in range(5):
        fast = engine.search(query_vector=query(seed), top_k=10)
        exact = engine.search(query_vector=query(seed), top_k=10, strategy="mask")
        assert fast.strategy == "hnsw" and ids(fast) == ids(exact)
        assert [r.excerpt_label for r in fast.results] == [r.excerpt_label for r in exact.results]


def test_auto_uses_exact_search_whenever_a_filter_is_set(engine):
    # Filtered HNSW lost recall in the measurements, even with a broad filter.
    broad = engine.search(filters=Filters(classification="B60B"), query_vector=query(1))
    narrow = engine.search(filters=Filters(classification="A61B"), query_vector=query(1))
    assert broad.strategy in ("prefilter", "mask") and narrow.strategy in ("prefilter", "mask")


def test_filters_stay_strict_with_hnsw(engine):
    report = engine.search(filters=Filters(classification="A61B"), query_vector=query(2),
                           top_k=10, strategy="hnsw")
    assert report.results and all(engine.patents[i]["classification"].startswith("A61B")
                                  for i in ids(report))
    exact = engine.search(filters=Filters(classification="A61B"), query_vector=query(2),
                          top_k=10, strategy="mask")
    assert ids(report) == ids(exact)


def test_similar_patent_search_never_returns_the_source_with_hnsw(engine):
    report, _ = engine.search_similar("3", top_k=10)
    assert report.strategy == "hnsw" and "3" not in ids(report) and len(ids(report)) == 10


def test_without_a_graph_auto_stays_exact_and_hnsw_is_refused(tmp_path):
    patents = [make_patent(str(i), "WHEEL", "A wheel.", "B60B100FI", ["1 . A wheel."])
               for i in range(5)]
    engine = build(tmp_path, patents, graph=False)
    assert engine.search(query_vector=query(0)).strategy == "no-filter"
    with pytest.raises(ValueError, match="no HNSW graph"):
        engine.search(query_vector=query(0), strategy="hnsw")


def test_a_graph_that_no_longer_matches_the_index_is_detected(tmp_path):
    patents = [make_patent(str(i), "WHEEL", "A wheel.", "B60B100FI", ["1 . A wheel."])
               for i in range(5)]
    engine = build(tmp_path, patents)
    build_hnsw(np.eye(16, dtype=np.float32)[:3], str(tmp_path / HNSW_FILE))   # wrong size
    with pytest.raises(ValueError, match="rebuild the index"):
        PatentSearch(str(tmp_path)).search(query_vector=query(0))
