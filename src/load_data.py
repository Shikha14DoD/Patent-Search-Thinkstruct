"""Load patent JSON files and turn them into cleaned, searchable units.

A "unit" is one piece of text that can be matched and shown as evidence:
an abstract, a single claim, or a paragraph of the detailed description.

Data-handling decisions, each driven by a measurement on the 640-patent
sample (see DECISIONS.md):

  D1  Patents are never dropped. 20,819 of 42,040 description paragraphs
      are blank, so blanks are skipped; the 119 patents left with no
      description text stay searchable through title, abstract and claims.

  D2  Claim numbers come from the printed identifier, never from the list
      position. 3,294 of 10,578 claims carry no printed number and 168
      entries are cancelled ranges ("1 - 5 . (canceled)"), so list position
      and real claim number diverge for 1,131 claims. Missing numbers are
      inferred from the surrounding sequence and flagged as inferred, so a
      result never shows a confident but wrong "Claim N".

  D3  The date in the file name is the A1 *publication* date, not the
      filing date. A1 documents publish roughly 18 months after filing, so
      the field is named publication_date and nothing claims to be a
      filing date.

  D4  Reference numerals are glued to neighbouring words in descriptions
      ("spoke100includes", "FIG.1is"): 95,905 occurrences, all in the
      detailed description (abstracts, titles and claims are clean). They
      are separated for both embedding and display, with measurement units
      ("205mm", "32psi") and codes ("B60B27/00", "3D") left intact.

  D5  Raw text is kept alongside the cleaned text. Cleaning is lossy, so
      the original is preserved for anyone who needs to check it.

  D6  Description paragraphs shorter than MIN_PARAGRAPH_WORDS are not
      indexed. 170 paragraphs are under five words (formula lines such as
      "B/A=1.00.", stray fragments such as "vehicle wheel."). A short query
      matches a short fragment almost perfectly, so a fragment with no
      technical content was outranking real evidence.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

# Roughly the number of words that fits the model's 256-token limit.
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 30
# Paragraphs shorter than this carry no usable evidence (see D6).
MIN_PARAGRAPH_WORDS = 5

# "1 . A spoke comprising" / "12. The spoke of claim 1"
CLAIM_NUMBER_RE = re.compile(r"^\s*(\d+)\s*\.\s*")
# "1 - 5 . (canceled)" — a cancelled range, not a claim
CANCELLED_RE = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*\.\s*\(?cancell?ed\)?", re.IGNORECASE)
CANCELLED_ONE_RE = re.compile(r"^\s*(\d+)\s*\.\s*\(?cancell?ed\)?", re.IGNORECASE)
# "of claim 1", "according to claim 3", "in claims 1 and 2"
CLAIM_REF_RE = re.compile(r"\bclaims?\s+(\d+)", re.IGNORECASE)

# -- reference-numeral separation (descriptions only) ---------------------
# Units that must stay attached to their number: "205mm", not "205 mm".
UNITS = {
    "mm", "cm", "m", "km", "nm", "um", "mg", "g", "kg", "s", "ms", "us", "ns",
    "h", "hr", "hz", "khz", "mhz", "ghz", "mpa", "kpa", "gpa", "pa", "psi",
    "bar", "n", "kn", "deg", "c", "f", "k", "v", "kv", "w", "kw", "a", "ma",
    "ml", "l", "in", "ft", "rpm", "mph", "kph", "wt", "vol", "d", "x",
    "st", "nd", "rd", "th", "b", "e", "sec", "min",
}
GLUED_FIGURE_RE = re.compile(r"([a-z.,;])(FIGS?\.|FIGURE\b|Figs?\.)")
FIGURE_NUMBER_RE = re.compile(r"\b(FIGS?|FIGURE|Figs?)\.(?=[0-9])")
LETTER_DIGIT_RE = re.compile(r"([a-z])(\d)")
DIGIT_LETTER_RE = re.compile(r"(\d)([A-Za-z]+)")


def _split_digit_letter(match: re.Match) -> str:
    digits, letters = match.group(1), match.group(2)
    if letters.lower() in UNITS:  # a measurement, leave it alone
        return f"{digits}{letters}"
    return f"{digits} {letters}"


def separate_reference_numerals(text: str) -> str:
    """Put spaces back around reference numerals in description text."""
    text = GLUED_FIGURE_RE.sub(r"\1 \2", text)
    text = FIGURE_NUMBER_RE.sub(r"\1. ", text)
    text = LETTER_DIGIT_RE.sub(r"\1 \2", text)
    text = DIGIT_LETTER_RE.sub(_split_digit_letter, text)
    return collapse_space(text)


def collapse_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass
class Patent:
    patent_id: str
    title: str
    abstract: str
    classification: str
    publication_date: str  # from the weekly file name; NOT the filing date
    url: str
    claims: list["Claim"] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)       # cleaned
    raw_paragraphs: list[str] = field(default_factory=list)   # untouched
    # Position of each kept paragraph among the non-blank paragraphs, so
    # "Description para 12" stays stable when short paragraphs are skipped.
    paragraph_numbers: list[int] = field(default_factory=list)


@dataclass
class Claim:
    number: int
    number_inferred: bool   # True when the printed number was missing
    preamble_missing: bool  # True when the text starts mid-sentence
    parent: int | None      # the claim this one depends on, if any
    text: str
    raw_text: str


@dataclass
class Unit:
    unit_id: int
    patent_id: str
    kind: str          # "abstract" | "claim" | "paragraph"
    position: int      # claim number, or paragraph number; 0 for the abstract
    label: str         # shown to the user, e.g. "Claim 4"
    label_note: str    # e.g. "claim number inferred"
    context_label: str # e.g. "Claim 6" when a parent claim's text was added
    text: str          # what gets embedded (may carry added context)
    display_text: str  # what the user sees
    raw_text: str      # the original, before any cleaning


def publication_date_from_filename(path: str) -> str:
    """patents_ipa240215.json -> 2024-02-15 (the A1 publication week)."""
    match = re.search(r"patents_ipa(\d{2})(\d{2})(\d{2})", os.path.basename(path))
    if not match:
        return ""
    yy, mm, dd = match.groups()
    return f"20{yy}-{mm}-{dd}"


def google_patents_url(filename: str, patent_id: str) -> str:
    """US20240051333A1-20240215.XML -> https://patents.google.com/patent/US20240051333A1"""
    stem = (filename or "").split("-")[0].strip()
    if not re.fullmatch(r"[A-Z]{2}\d+[A-Z]\d?", stem):
        stem = f"US{patent_id}A1"
    return f"https://patents.google.com/patent/{stem}"


def parse_claims(raw_claims: list[str]) -> list[Claim]:
    """Parse one patent's claims, recovering the real claim numbers.

    Walks the list in order, tracking the highest number seen so far:
      - "7 . The tire of claim 6"  -> printed number 7
      - "1 - 5 . (canceled)"       -> not a claim; advances the counter to 5
      - unnumbered text            -> inferred as previous + 1, flagged
    """
    claims: list[Claim] = []
    last_number = 0

    for raw in raw_claims:
        stripped = collapse_space(raw)
        if not stripped:
            continue

        cancelled = CANCELLED_RE.match(stripped) or CANCELLED_ONE_RE.match(stripped)
        if cancelled:
            # Cancelled claims carry no technical content, so they are not
            # indexed; they only tell us where the numbering resumes.
            last_number = int(cancelled.group(cancelled.re.groups))
            continue

        numbered = CLAIM_NUMBER_RE.match(stripped)
        if numbered:
            number = int(numbered.group(1))
            text = stripped[numbered.end():]
            inferred = False
        else:
            number = last_number + 1
            text = stripped
            inferred = True

        if not text:
            continue
        last_number = max(last_number, number)

        parent_match = CLAIM_REF_RE.search(text)
        parent = int(parent_match.group(1)) if parent_match else None
        if parent == number:
            parent = None

        # An independent claim normally opens "A tire comprising:". When the
        # text starts mid-sentence the preamble was lost in conversion.
        preamble_missing = not re.match(r"^(A|An|The)\b", text)

        claims.append(Claim(number=number, number_inferred=inferred,
                            preamble_missing=preamble_missing, parent=parent,
                            text=text, raw_text=raw.strip()))
    return claims


def chunk_paragraph(text: str) -> list[str]:
    """Split a long paragraph into overlapping word chunks."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]
    chunks: list[str] = []
    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    for start in range(0, len(words), step):
        chunk = words[start:start + CHUNK_WORDS]
        if len(chunk) < 20 and chunks:  # don't emit a tiny trailing fragment
            break
        chunks.append(" ".join(chunk))
    return chunks


def load_patents(data_dir: str) -> tuple[list[Patent], dict[str, int]]:
    """Read every patents_ipa*.json file in data_dir. Returns (patents, report)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No .json files found in {data_dir!r}. "
            "Unzip the dataset so the files sit directly inside that folder."
        )

    patents: list[Patent] = []
    seen: set[str] = set()
    report = {"files": len(paths), "records": 0, "skipped_no_id": 0,
              "skipped_duplicate_id": 0, "blank_paragraphs": 0,
              "short_paragraphs_skipped": 0,
              "cancelled_claim_entries": 0, "claims_number_inferred": 0,
              "claims_preamble_missing": 0}

    for path in paths:
        published = publication_date_from_filename(path)
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            report["records"] += 1
            patent_id = str(record.get("doc_number", "")).strip()
            if not patent_id:
                report["skipped_no_id"] += 1
                continue
            if patent_id in seen:
                report["skipped_duplicate_id"] += 1
                continue
            seen.add(patent_id)

            non_blank = [p for p in record.get("detailed_description") or []
                         if p and p.strip()]
            report["blank_paragraphs"] += (
                len(record.get("detailed_description") or []) - len(non_blank))
            kept = [(number, p) for number, p in enumerate(non_blank, start=1)
                    if len(p.split()) >= MIN_PARAGRAPH_WORDS]
            raw_paragraphs = [p for _, p in kept]
            report["short_paragraphs_skipped"] += len(non_blank) - len(kept)

            raw_claims = record.get("claims") or []
            report["cancelled_claim_entries"] += sum(
                1 for c in raw_claims
                if CANCELLED_RE.match(collapse_space(c)) or CANCELLED_ONE_RE.match(collapse_space(c)))
            claims = parse_claims(raw_claims)
            report["claims_number_inferred"] += sum(1 for c in claims if c.number_inferred)
            report["claims_preamble_missing"] += sum(1 for c in claims if c.preamble_missing)

            patents.append(Patent(
                patent_id=patent_id,
                title=collapse_space(record.get("title")),
                abstract=collapse_space(record.get("abstract")),
                classification=(record.get("classification") or "").strip(),
                publication_date=published,
                url=google_patents_url(record.get("filename", ""), patent_id),
                claims=claims,
                paragraphs=[separate_reference_numerals(p) for p in raw_paragraphs],
                raw_paragraphs=[collapse_space(p) for p in raw_paragraphs],
                paragraph_numbers=[number for number, _ in kept],
            ))
    return patents, report


def build_units(patents: list[Patent]) -> list[Unit]:
    """Turn patents into the flat list of searchable units."""
    units: list[Unit] = []
    next_id = 0

    for patent in patents:
        if patent.abstract:
            # The title is short but informative, so it rides along with the
            # abstract for embedding rather than getting a vector of its own.
            text = f"{patent.title}. {patent.abstract}" if patent.title else patent.abstract
            units.append(Unit(next_id, patent.patent_id, "abstract", 0, "Abstract", "", "",
                              text, patent.abstract, patent.abstract))
            next_id += 1

        by_number = {claim.number: claim.text for claim in patent.claims}
        for claim in patent.claims:
            # A dependent claim is a fragment on its own, and a claim whose
            # preamble was lost starts mid-sentence. Both get context for the
            # embedding only; the user still sees the claim as published.
            context: list[str] = []
            context_label = ""
            if claim.parent and claim.parent in by_number:
                context.append(by_number[claim.parent])
                context_label = f"Claim {claim.parent}"
            elif claim.preamble_missing and patent.title:
                context.append(patent.title)
            embed_text = " ".join(context + [claim.text]) if context else claim.text

            notes = []
            if claim.number_inferred:
                notes.append("claim number inferred")
            if claim.preamble_missing:
                notes.append("opening text missing in source")

            units.append(Unit(next_id, patent.patent_id, "claim", claim.number,
                              f"Claim {claim.number}", "; ".join(notes), context_label,
                              embed_text, claim.text, claim.raw_text))
            next_id += 1

        for index, cleaned, raw in zip(
                patent.paragraph_numbers, patent.paragraphs, patent.raw_paragraphs):
            for chunk_number, chunk in enumerate(chunk_paragraph(cleaned)):
                label = f"Description para {index}"
                if chunk_number:
                    label += f" (part {chunk_number + 1})"
                units.append(Unit(next_id, patent.patent_id, "paragraph", index,
                                  label, "", "", chunk, chunk,
                                  raw if chunk_number == 0 else ""))
                next_id += 1

    return units


def load_corpus(data_dir: str) -> tuple[list[Patent], list[Unit], dict[str, int]]:
    patents, report = load_patents(data_dir)
    return patents, build_units(patents), report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the cleaned corpus.")
    parser.add_argument("--data", default="data/patent_data_small")
    parser.add_argument("--show", default="", help="Print one patent's claims by id")
    args = parser.parse_args()

    patents, units, report = load_corpus(args.data)
    kinds: dict[str, int] = {}
    for unit in units:
        kinds[unit.kind] = kinds.get(unit.kind, 0) + 1

    print(f"patents: {len(patents)}")
    print(f"units:   {len(units)}  {kinds}")
    print("ingest report:")
    for key, value in report.items():
        print(f"  {key:<28} {value}")
    print(f"  {'patents_without_description':<28} "
          f"{sum(1 for p in patents if not p.paragraphs)}")

    if args.show:
        patent = next(p for p in patents if p.patent_id == args.show)
        print(f"\n{patent.patent_id} {patent.title} ({patent.classification}, "
              f"published {patent.publication_date})")
        for claim in patent.claims:
            flags = []
            if claim.number_inferred:
                flags.append("inferred number")
            if claim.preamble_missing:
                flags.append("preamble missing")
            if claim.parent:
                flags.append(f"depends on {claim.parent}")
            print(f"  Claim {claim.number:<3} [{', '.join(flags) or 'ok'}] "
                  f"{claim.text[:80]}")