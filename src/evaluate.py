"""OCR metric, unseen-brand test, selective prediction and report figures.

    python -m src.evaluate          # after src.train

Adds to outputs/metrics.json and writes reports/figures/*.png.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from .features import FusionClassifier, build_tfidf_lr, lexicon_flags  # noqa: E402
from .lexicon import Lexicon  # noqa: E402
from .ocr import load_image  # noqa: E402
from .text_normalize import normalize  # noqa: E402
from .train import VIEW_SETS, metrics, oof_proba, with_text  # noqa: E402

FIG = Path("reports/figures")
BLUE, LIGHT_BLUE, ORANGE = "#2a78d6", "#86b6ef", "#eb6834"


def ocr_term_recall(df: pd.DataFrame) -> dict:
    """Intermediate OCR metric (no transcripts available): share of alcohol images where the
    lexicon finds an alcohol term, and of non-alcohol images where it finds a decoy."""
    y = df["label"].to_numpy() == 1
    out = {}
    for name, views in VIEW_SETS.items():
        alc, neg = lexicon_flags(with_text(df, views))
        out[name] = {"alcohol_term_found_on_alcohol": round(float(alc[y].mean()), 4),
                     "decoy_found_on_non_alcohol": round(float(neg[~y].mean()), 4),
                     "alcohol_term_found_on_non_alcohol": round(float(alc[~y].mean()), 4)}
    return out


def unseen_brands(df: pd.DataFrame, C: float) -> dict:
    """GroupKFold where all images mentioning the same beverage term share a fold, so every
    test fold only contains drink/brand names never seen in training."""
    X, y, lex = with_text(df), df["label"].to_numpy(), Lexicon()
    groups = []
    for i, t in enumerate(X["text"]):
        m = lex.match(t)
        groups.append(min(m.alcohol + m.negative) if m.alcohol or m.negative else f"none_{i}")

    class GroupCV:
        def split(self, X_, y_):
            return GroupKFold(n_splits=5).split(X_, y_, groups)

    return {"TF-IDF logistic regression": metrics(y, oof_proba(build_tfidf_lr(C), X, y, GroupCV())),
            "fusion (final)": metrics(y, oof_proba(FusionClassifier(C), X, y, GroupCV()))}


def selective_prediction(pred: pd.DataFrame, thr: float) -> list[dict]:
    """Accuracy when predictions inside an abstain band around the threshold go to review."""
    rows = []
    for band in [0.0, 0.05, 0.1, 0.15]:
        sure = (pred["prob"] < thr - band) | (pred["prob"] >= thr + band)
        rows.append({"abstain_band": [round(thr - band, 2), round(thr + band, 2)] if band else None,
                     "auto_decided": round(float(sure.mean()), 3),
                     "accuracy_on_auto_decided": round(float((pred["pred"] == pred["label"])[sure].mean()), 3)})
    return rows


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_ocr(recall: dict) -> None:
    """Bar chart of beverage-term recall per OCR configuration."""
    names, vals = list(recall)[::-1], [recall[k]["alcohol_term_found_on_alcohol"] for k in list(recall)[::-1]]
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    bars = ax.barh(names, vals, height=0.6, color=[BLUE if n.startswith("final") else LIGHT_BLUE for n in names])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of alcohol images where the beverage term is read")
    ax.set_title("OCR configuration vs beverage-term recall", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "ocr_views.png", dpi=110)
    plt.close(fig)


def plot_results(pred: pd.DataFrame) -> None:
    """Confusion matrices, ROC and precision-recall curves for CV and hold-out."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for i, (name, split) in enumerate([("CV out-of-fold (880)", "cv"), ("Hold-out (220)", "holdout")]):
        p = pred[pred["split"] == split]
        ConfusionMatrixDisplay.from_predictions(p["label"], p["pred"], display_labels=["non_alcohol", "alcohol"],
                                                ax=ax[i], colorbar=False, cmap="Blues")
        ax[i].set_title(name)
        RocCurveDisplay.from_predictions(p["label"], p["prob"], ax=ax[2], name=name, curve_kwargs={"color": [BLUE, ORANGE][i]})
        PrecisionRecallDisplay.from_predictions(p["label"], p["prob"], ax=ax[3], name=name, color=[BLUE, ORANGE][i])
    ax[2].set_title("ROC")
    ax[3].set_title("Precision-Recall")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_roc_pr.png", dpi=110)
    plt.close(fig)


def plot_failures(df: pd.DataFrame, pred: pd.DataFrame, n: int = 8) -> None:
    """Gallery of the n most confident errors with their OCR text."""
    d = df.merge(pred, on=["file_name", "label"])
    wrong = d[d["pred"] != d["label"]].assign(margin=lambda x: (x["prob"] - 0.5).abs())
    wrong = wrong.sort_values("margin", ascending=False).head(n)
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        ax.imshow(load_image(Path(r["path"])))
        truth = "alcohol" if r["label"] == 1 else "non_alcohol"
        ax.set_title(f"{r['file_name']}  true={truth}  p={r['prob']:.2f}", fontsize=9)
        ocr = textwrap.shorten(normalize(f"{r['text_pm_orig']} {r['text_pm_flip']}"), 100)
        ax.text(0, -0.02, "\n".join(textwrap.wrap(f"OCR: {ocr}", 50)), transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "failure_cases.png", dpi=110)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("outputs/ocr_dataset.csv", keep_default_na=False)
    m = json.load(open("outputs/metrics.json"))
    pred = pd.read_csv("outputs/predictions.csv")
    pred["pred"] = (pred["prob"] >= m["threshold"]).astype(int)

    m["ocr_term_recall"] = ocr_term_recall(df)
    m["unseen_brands"] = unseen_brands(df, m["C"])
    m["selective_prediction"] = selective_prediction(pred, m["threshold"])
    json.dump(m, open("outputs/metrics.json", "w"), indent=2)

    plot_ocr(m["ocr_term_recall"])
    plot_results(pred)
    plot_failures(df, pred)
    print(json.dumps({k: m[k] for k in ("ocr_term_recall", "unseen_brands", "selective_prediction")}, indent=1))


if __name__ == "__main__":
    main()
