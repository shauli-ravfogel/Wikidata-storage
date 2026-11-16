"""Text normalization shared by entity lookup and answer grading.

Two keys are used for names:
  * `norm(s)`      lowercase, accents stripped, punctuation removed, whitespace collapsed
  * `sorted_key(s)` the same tokens sorted, so word-order variants match
                    ("Pernambucano Juninho" == "Juninho Pernambucano")
Stdlib only, so it runs anywhere.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")
_YEAR = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
# Words that carry no identity in names/answers ("The Beatles" vs "Beatles").
_STOPWORDS = {"the", "a", "an", "of", "de", "da", "do", "dos", "das", "von", "van", "la", "le"}


def strip_accents(s: str) -> str:
    """'São Paulo' -> 'Sao Paulo'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s: str) -> str:
    """Canonical form for exact matching: 'São Paulo, Brazil!' -> 'sao paulo brazil'."""
    s = strip_accents(s).lower()
    s = _PUNCT.sub(" ", s)
    return _SPACE.sub(" ", s).strip()


def tokens(s: str) -> list[str]:
    """Normalized tokens without stopwords."""
    return [t for t in norm(s).split() if t not in _STOPWORDS]


def sorted_key(s: str) -> str:
    """Order-insensitive key: 'Pernambucano, Juninho' -> 'juninho pernambucano'."""
    return " ".join(sorted(tokens(s)))


def extract_year(s: str) -> str | None:
    """First 3-4 digit number in the string ('born in 1974.' -> '1974'), else None."""
    m = _YEAR.search(s)
    return m.group(1) if m else None
