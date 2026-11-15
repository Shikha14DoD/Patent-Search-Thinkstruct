"""Tests for the BM25 keyword baseline, on a tiny hand-made index (no model)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_search import make_engine, make_patent  # noqa: E402

from src.bm25 import BM25  # noqa: E402
from src.search import Filters  # noqa: E402


def engine_with(tmp_path, abstracts):
    patents = [make_patent(str(i), f"PATENT {i}", text, code, ["1 . A part."])
               for i, (text, code) in enumerate(abstracts, start=1)]
    return make_engine(tmp_path, patents, {})


def test_more_query_words_ranks_higher(tmp_path):
    engine = engine_with(tmp_path, [("A tire with a pressure sensor.", "B60C"),
                                    ("A tire with a tread.", "B60C"),
                                    ("A trailer hitch.", "B60D")])
    ranked = BM25(engine).search("tire pressure sensor")
    assert [patent_id for patent_id, _, _ in ranked] == ["1", "2"]   # 3 shares no word


def test_rare_words_count_more_than_common_ones(tmp_path):
    # "tire" appears everywhere, "sensor" once: the patent with "sensor" must win.
    engine = engine_with(tmp_path, [("A tire tire tire.", "B60C"),
                                    ("A tire with a sensor.", "B60C"),
                                    ("A tire rim.", "B60C"),
                                    ("A tire tread.", "B60C")])
    ranked = BM25(engine).search("tire sensor")
    assert ranked[0][0] == "2"


def test_uses_the_same_stemming_and_filters_as_search(tmp_path):
    engine = engine_with(tmp_path, [("Pressure sensors in tyres.", "B60C"),
                                    ("A sensor in a hitch.", "B60D")])
    bm25 = BM25(engine)
    assert bm25.search("tire sensor")[0][0] == "1"                     # tyres = tire, sensors = sensor
    assert [p for p, _, _ in bm25.search("sensor", Filters(classification="B60D"))] == ["2"]


def test_no_shared_word_means_no_result(tmp_path):
    engine = engine_with(tmp_path, [("A tire.", "B60C")])
    assert BM25(engine).search("battery cooling") == []