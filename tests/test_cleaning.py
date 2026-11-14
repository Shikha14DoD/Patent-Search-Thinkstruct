"""Tests for the data-cleaning rules. None of them need the model or the index.

Run with:  python -m pytest tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_data import (  # noqa: E402
    TOKEN_BUDGET,
    Patent,
    TokenCounter,
    build_units,
    chunk_text,
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


# -- fitting the model's input limit (D7) ------------------------------------
# Without a tokenizer, TokenCounter estimates one token per four characters,
# so "word" costs 1 token: these tests control token counts exactly.
def test_long_text_is_split_with_overlap():
    counter = TokenCounter()
    text = " ".join(["word"] * 600)
    parts = chunk_text(text, counter, budget=250, overlap=32)
    assert all(counter.count(part) <= 250 for part in parts)
    assert len(parts) == 3
    assert parts[0].split()[-32:] == parts[1].split()[:32]  # overlap
    assert chunk_text("a short paragraph", counter) == ["a short paragraph"]


def _patent_with_claims(claims):
    from src.load_data import parse_claims
    return Patent(patent_id="1", title="WHEEL", abstract="A wheel.", classification="B60B",
                  publication_date="2024-01-01", url="u", claims=parse_claims(claims))


def test_dependent_claim_keeps_its_own_text_when_the_parent_is_long():
    parent = "1 . A wheel comprising " + "word " * 300
    child_own = "The wheel of claim 1 , wherein the spoke is hollow and made of steel."
    patent = _patent_with_claims([parent, "2 . " + child_own])
    units, stats = build_units([patent], TokenCounter())
    child = [u for u in units if u.label == "Claim 2"][0]
    counter = TokenCounter()
    assert child.text.endswith(child_own)              # own text intact, at the end
    assert counter.count(child.text) <= TOKEN_BUDGET   # whole text fits
    assert child.context_label == "Claim 1"
    assert stats["parent_context_trimmed"] == 1


def test_every_unit_fits_the_budget():
    long_claim = "1 . A wheel comprising " + "word " * 700
    patent = _patent_with_claims([long_claim, "2 . The wheel of claim 1 , wherein it is red."])
    units, stats = build_units([patent], TokenCounter())
    assert all(TokenCounter().count(u.text) <= TOKEN_BUDGET for u in units)
    assert [u.label for u in units if u.position == 1][:2] == ["Claim 1 (part 1)",
                                                               "Claim 1 (part 2)"]


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