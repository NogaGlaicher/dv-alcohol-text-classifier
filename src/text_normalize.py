"""Text cleaning tailored to OCR output of adversarial overlay text.

Handles: random casing ("nJoy taSTe"), leetspeak ("br3w", "v0dka", "w!ne"), punctuation
noise from OCR, repeated phrases, and words split by OCR ("bour bon").
"""
from __future__ import annotations

import re
import unicodedata

# Digits/symbols commonly used as letter substitutes, and common OCR confusions.
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
    "@": "a", "$": "s", "!": "i", "|": "l", "€": "e", "£": "l",
})

_non_alpha = re.compile(r"[^a-z\s]+")
_spaces = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    """Unicode-fold and lowercase, keep symbols (used for the 'raw' char n-gram view)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _spaces.sub(" ", text.lower()).strip()


def normalize(text: str) -> str:
    """Lowercase, de-leet, keep letters only, collapse whitespace."""
    text = basic_clean(text).translate(LEET_MAP)
    text = text.replace("'", "")           # don't -> dont
    text = _non_alpha.sub(" ", text)
    return _spaces.sub(" ", text).strip()


def dedupe_tokens(text: str) -> str:
    """Drop immediately repeated tokens / bigrams ('limited edition limited edition')."""
    out: list[str] = []
    toks = text.split()
    i = 0
    while i < len(toks):
        if out and out[-1] == toks[i]:                       # repeated unigram
            i += 1
        elif len(out) >= 2 and out[-2:] == toks[i:i + 2]:    # repeated bigram
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def spaceless(text: str) -> str:
    return text.replace(" ", "")


# Glyph sequences OCR engines commonly confuse; applied to BOTH the OCR token and the
# lexicon term so "smimoff" (OCR) matches "smirnoff".
OCR_CONFUSIONS = [("rn", "m"), ("vv", "w"), ("cl", "d")]


def ocr_canon(word: str) -> str:
    for a, b in OCR_CONFUSIONS:
        word = word.replace(a, b)
    return word
