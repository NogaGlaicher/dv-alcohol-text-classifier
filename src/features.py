"""The classifier: TF-IDF logistic regression + domain-lexicon late fusion.

Input: a DataFrame with a `text` column (OCR text of all views concatenated).

1. Text model. Normalise (lowercase, de-leet, de-duplicate), drop template words
   (lexicon.TEMPLATE_WORDS) and 1-letter OCR debris, then TF-IDF char 2-5-grams + word 1-2-grams
   -> class-balanced logistic regression. Char n-grams tolerate OCR noise (" cha", "ine", "bud").
2. Late fusion with the lexicon, which also covers drinks and brands absent from training:
       alcohol term found, no decoy  -> p = max(p, 0.95)
       decoy found, no alcohol term  -> p = min(p, 0.05)
       otherwise                     -> p unchanged
   Late rather than as LR features: the lexicon hit is ~99.7% precise, so an LR given it as a
   feature learns "no hit => non-alcohol" and loses ~13 points of recall (see report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from .lexicon import TEMPLATE_WORDS, Lexicon
from .text_normalize import dedupe_tokens, normalize

SEED = 42
LEX_POS, LEX_NEG = 0.95, 0.05


def _texts(X) -> pd.Series:
    s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.ravel(X))
    return s.fillna("").astype(str)


def clean_text(X) -> list[str]:
    """Normalised content words only (template words and 1-letter OCR debris removed)."""
    return [" ".join(w for w in dedupe_tokens(normalize(t)).split() if len(w) > 1 and w not in TEMPLATE_WORDS)
            for t in _texts(X)]


def build_tfidf_lr(C: float = 1.0) -> Pipeline:
    """Text model: cleaned text -> TF-IDF char 2-5 + word 1-2 grams -> balanced L2 logistic regression."""
    clean = FunctionTransformer(clean_text, feature_names_out="one-to-one")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
    features = ColumnTransformer([("char", make_pipeline(clean, char), ["text"]),
                                  ("word", make_pipeline(clean, word), ["text"])])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED))])


def lexicon_flags(X) -> tuple[np.ndarray, np.ndarray]:
    """(alcohol term found, decoy term found) per text."""
    lex = Lexicon()
    ms = [lex.match(t) for t in _texts(X)]
    return np.array([bool(m.alcohol) for m in ms]), np.array([bool(m.negative) for m in ms])


class FusionClassifier(BaseEstimator, ClassifierMixin):
    """TF-IDF logistic regression with lexicon late fusion."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X, y):
        self.model_ = build_tfidf_lr(self.C).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X)[:, 1]
        alc, neg = lexicon_flags(X)
        p = np.where(alc & ~neg, np.maximum(p, LEX_POS), p)
        p = np.where(neg & ~alc, np.minimum(p, LEX_NEG), p)
        return np.column_stack([1 - p, p])
