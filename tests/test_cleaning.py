"""Tests for the data-cleaning rules. None of them need the model or the index.

Run with:  python -m pytest tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_data import (  # noqa: E402
    chunk_paragraph,
    google_patents_url,
    parse_claims,
    publication_date_from_filename,
    separate_reference_numerals,
)
from src.text_utils import normalize_title, parse_keywords, tokenize  # noqa: E402


# -- claim numbering (D2) ---------------------------------------------------
def test_cancelled_range_advances_the_numbering():
    # Shape taken from patent 20240051338 in the sample data.
    claims = parse_claims([
        "1 - 5 . (canceled)",
        "a tread portion extending in a tire circumferential direction",
        "7 . The pneumatic tire according to claim 6 , wherein the rubber is soft.",
        "8 . The pneumatic tire according to claim 6 , wherein the rubber is hard.",
        "the undertread layer contains a rubber composition",
        "10 . The pneumatic tire according to claim 9 , wherein the ratio is 2.",
    ])
    assert [c.number for c in claims] == [6, 7, 8, 9, 10]
    assert [c.number_inferred for c in claims] == [True, False, False, True, False]
    assert [c.parent for c in claims] == [None, 6, 6, None, 9]


def test_cancelled_entries_are_not_claims():
    claims = parse_claims(["1 . A wheel comprising a rim.", "2 . (canceled)",
                           "3 . The wheel of claim 1 , wherein the rim is steel."])
    assert [c.number for c in claims] == [1, 3]


def test_printed_number_is_stripped_from_the_text():
    (claim,) = parse_claims(["12 . The spoke of claim 1 , wherein the body is hollow."])
    assert claim.number == 12
    assert claim.text.startswith("The spoke of claim 1")
    assert claim.raw_text.startswith("12 .")  # the original is kept


def test_missing_preamble_is_flagged():
    first, second = parse_claims(["an axle body, having a middle segment",
                                  "2 . The spoke of claim 1 , wherein it is long."])
    assert first.preamble_missing and first.number == 1 and first.number_inferred
    assert not second.preamble_missing


# -- reference numerals (D4) ------------------------------------------------
def test_glued_reference_numerals_are_separated():
    assert separate_reference_numerals(
        "The spoke100includes an axle body110and two connecting elements120."
    ) == "The spoke 100 includes an axle body 110 and two connecting elements 120."


def test_glued_figure_references_are_separated():
    assert separate_reference_numerals(
        "an embodiment.FIG.2is a diagram, described with reference toFIGS.1to12below."
    ) == "an embodiment. FIG. 2 is a diagram, described with reference to FIGS. 1 to 12 below."


def test_units_and_codes_are_left_alone():
    text = "a width of 205mm at 32psi and 3.5MPa, class B60B27/00, a 3D part at 1500rpm"
    assert separate_reference_numerals(text) == text


# -- other loading rules ----------------------------------------------------
def test_publication_date_and_url():
    assert publication_date_from_filename("x/patents_ipa240215.json") == "2024-02-15"
    assert google_patents_url("US20240051333A1-20240215.XML", "20240051333") == (
        "https://patents.google.com/patent/US20240051333A1")
    assert google_patents_url("", "20240051333").endswith("/US20240051333A1")


def test_long_paragraphs_are_chunked_with_overlap():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_paragraph(" ".join(words))
    assert len(chunks) == 3
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]  # overlap
    assert chunk_paragraph("a short paragraph") == ["a short paragraph"]


# -- text normalisation -----------------------------------------------------
def test_singular_and_plural_share_a_stem():
    assert tokenize("Sensors") == tokenize("sensor")
    assert tokenize("tyres") == tokenize("tire")
    assert tokenize("braking") == tokenize("brake")


def test_keywords_and_phrases():
    terms = parse_keywords('"wheel hub" sensors')
    assert [tokens for _, tokens in terms] == [["wheel", "hub"], ["sensor"]]


def test_title_normalisation_ignores_case_and_punctuation():
    assert normalize_title("Pneumatic  Tire") == normalize_title("PNEUMATIC TIRE")