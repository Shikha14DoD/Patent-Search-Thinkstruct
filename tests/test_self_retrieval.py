"""Tests for the self-retrieval check's ranking, on a tiny hand-made index."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_search import make_engine, make_patent  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from self_retrieval import first_claim_rows, target_rank  # noqa: E402


def test_own_claims_are_hidden_when_ranking(tmp_path):
    patents = [make_patent("1", "A", "First abstract.", "B60B", ["1 . A wheel.", "2 . The wheel of claim 1 ."]),
               make_patent("2", "B", "Second abstract.", "B60B", ["1 . A tire."])]
    engine = make_engine(tmp_path, patents, {
        ("1", "Claim 1"): 0.99,       # would win, but it is the query patent's own claim
        ("1", "Abstract"): 0.40,
        ("2", "Abstract"): 0.50})
    scores = engine.vectors @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert target_rank(engine, scores, "1") == 2        # patent 2's abstract now beats it
    # Searching for patent 2, only patent 2's claims are hidden: patent 1's
    # claim stays visible and outranks it.
    assert target_rank(engine, scores, "2") == 2


def test_bm25_zero_score_is_a_miss(tmp_path):
    patents = [make_patent("1", "A", "Abstract.", "B60B", ["1 . A wheel."])]
    engine = make_engine(tmp_path, patents, {})
    zeros = np.zeros(len(engine.units), dtype=np.float32)
    assert target_rank(engine, zeros, "1", zero_is_miss=True) is None


def test_first_claim_is_the_lowest_numbered(tmp_path):
    patents = [make_patent("1", "A", "Abstract.", "B60B",
                           ["1 - 2 . (canceled)", "3 . A wheel.", "4 . The wheel of claim 3 ."])]
    engine = make_engine(tmp_path, patents, {})
    assert engine.units[first_claim_rows(engine)["1"]]["label"] == "Claim 3"