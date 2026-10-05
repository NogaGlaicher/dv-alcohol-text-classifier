"""Model selection, comparisons, hold-out evaluation and the final model.

    python -m src.train

* Stratified 80/20 split (seed 42); the 20% hold-out is only used for reporting.
* 5-fold CV on the 80% (same folds everywhere): choose C and the decision threshold, and
  compare (a) models on the final OCR text and (b) OCR view sets with the final model.
* The final model is refit on all data -> models/model.joblib.
Outputs: outputs/metrics.json, outputs/predictions.csv (out-of-fold + hold-out probabilities).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features import SEED, FusionClassifier, build_tfidf_lr, lexicon_flags
from .ocr import EASYOCR_VIEWS, FINAL_VIEWS, PADDLE_VIEWS

C_GRID = [1, 2, 4, 8, 16, 32]
VIEW_SETS = {
    "EasyOCR, original image": ["orig"],
    "EasyOCR, 4 views": list(EASYOCR_VIEWS),
    "Paddle, 4 views": list(PADDLE_VIEWS),
    "final: EasyOCR orig + Paddle 4 views": list(FINAL_VIEWS),
}


def metrics(y, prob, thr=0.5) -> dict:
    """Precision, recall, F1, accuracy (and ROC-AUC for probabilities) at threshold `thr`."""
    pred = (np.asarray(prob) >= thr).astype(int)
    out = {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
           "f1": f1_score(y, pred), "accuracy": float((pred == y).mean())}
    if len(np.unique(prob)) > 2:
        out["roc_auc"] = roc_auc_score(y, prob)
    return {k: round(float(v), 4) for k, v in out.items()}


def with_text(df: pd.DataFrame, views=FINAL_VIEWS) -> pd.DataFrame:
    """Model input: the texts of the given OCR views concatenated."""
    return df[[f"text_{v}" for v in views]].fillna("").astype(str).agg(" ".join, axis=1).to_frame("text")


def oof_proba(model, X: pd.DataFrame, y: np.ndarray, cv) -> np.ndarray:
    """Out-of-fold probabilities: every row is predicted by a model that did not train on it."""
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        oof[va] = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return oof


def best_threshold(y, prob) -> float:
    """Threshold maximising F1; among ties the one closest to 0.5."""
    grid = np.round(np.arange(0.2, 0.81, 0.01), 2)
    f1s = np.array([f1_score(y, (prob >= t).astype(int)) for t in grid])
    cands = grid[f1s >= f1s.max() - 1e-9]
    return float(cands[np.argmin(np.abs(cands - 0.5))])


def main() -> None:
    df = pd.read_csv("outputs/ocr_dataset.csv", keep_default_na=False)
    y_all = df["label"].to_numpy()
    dev, hold = train_test_split(df, test_size=0.2, stratify=y_all, random_state=SEED)
    dev, hold = dev.reset_index(drop=True), hold.reset_index(drop=True)
    y, y_hold = dev["label"].to_numpy(), hold["label"].to_numpy()
    X, X_hold = with_text(dev), with_text(hold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # C: the strongest regularisation within 0.005 F1 of the best (smaller differences are noise)
    c_f1 = {C: metrics(y, oof_proba(build_tfidf_lr(C), X, y, cv))["f1"] for C in C_GRID}
    C = min(c for c in C_GRID if c_f1[c] >= max(c_f1.values()) - 0.005)
    oof = oof_proba(FusionClassifier(C), X, y, cv)
    thr = best_threshold(y, oof)
    print(f"C={C} threshold={thr}")

    models = {
        "lexicon rule (alcohol term found)": metrics(y, lexicon_flags(X)[0].astype(float)),
        "TF-IDF logistic regression": metrics(y, oof_proba(build_tfidf_lr(C), X, y, cv)),
        "fusion (final), threshold 0.5": metrics(y, oof),
        f"fusion (final), threshold {thr}": metrics(y, oof, thr),
    }
    views = {name: metrics(y, oof_proba(FusionClassifier(C), with_text(dev, vs), y, cv))
             for name, vs in VIEW_SETS.items()}
    for name, m in {**models, **views}.items():
        print(f"{name:<45} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    p_hold = FusionClassifier(C).fit(X, y).predict_proba(X_hold)[:, 1]
    holdout = metrics(y_hold, p_hold, thr)
    print("hold-out:", holdout)

    final = FusionClassifier(C).fit(with_text(df), y_all)
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "threshold": thr, "views": list(FINAL_VIEWS)}, "models/model.joblib")

    pd.concat([dev[["file_name", "label"]].assign(split="cv", prob=oof),
               hold[["file_name", "label"]].assign(split="holdout", prob=p_hold)]
              ).to_csv("outputs/predictions.csv", index=False)
    json.dump({"C": C, "threshold": thr, "cv_models": models, "cv_ocr_views": views,
               "cv": models[f"fusion (final), threshold {thr}"], "holdout": holdout},
              open("outputs/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
