"""Which embedding model the index uses, pinned to one exact version.

A model on the Hugging Face Hub can be updated under the same name, which would
quietly change every vector and every number in the README. Pinning the commit
keeps builds and searches reproducible: the same text always gets the same
vector (the same reasoning as Part 2's versioned indexes).
"""

from __future__ import annotations

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The commit the README's results were produced with.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def revision_for(model_name: str) -> str | None:
    """The pinned commit for the default model; other models load their latest version."""
    return MODEL_REVISION if model_name == DEFAULT_MODEL else None
