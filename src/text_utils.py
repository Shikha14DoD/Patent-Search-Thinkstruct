"""Tokenising and normalising, shared by index building and query handling.

Both sides must normalise words the same way, or a filter typed by the user
will not match the index. Keeping it in one module guarantees that.
"""

from __future__ import annotations

import re

import snowballstemmer

TOKEN_RE = re.compile(r"[a-z0-9]+")

_STEMMER = snowballstemmer.stemmer("english")

# British/American variants that appear in this dataset ("TYRE" vs "TIRE").
# Applied before stemming so both spellings land on the same index entry.
SPELLING_VARIANTS = {
    "tyre": "tire",
    "tyres": "tires",
    "fibre": "fiber",
    "fibres": "fibers",
    "vulcanised": "vulcanized",
    "colour": "color",
    "colours": "colors",
    "centre": "center",
    "centres": "centers",
}


def stem(word: str) -> str:
    """Normalise a word so singular and plural forms match.

    Uses the Snowball (Porter2) stemmer: sensors -> sensor, braking -> brake.
    Stems are not always real words ("pressure" -> "pressur"), which is fine:
    they only have to be consistent between the index and the query.
    """
    return _STEMMER.stemWord(SPELLING_VARIANTS.get(word, word))


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, then stem."""
    return [stem(token) for token in TOKEN_RE.findall((text or "").lower())]


def normalize_title(title: str) -> str:
    """Normalise a title for exact lookup: case, punctuation and spacing."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", (title or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_keywords(raw: str) -> list[tuple[str, list[str]]]:
    """Split a filter box into terms.

    Quoted text stays together as a phrase; everything else is a single word.
    Returns (display form, normalised tokens) pairs.
    """
    if not raw:
        return []
    terms: list[tuple[str, list[str]]] = []
    for quoted, bare in re.findall(r'"([^"]+)"|(\S+)', raw):
        display = (quoted or bare).strip()
        tokens = tokenize(display)
        if tokens:
            terms.append((display, tokens))
    return terms