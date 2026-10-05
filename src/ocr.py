"""OCR stage: two CPU OCR engines x several image "views" (test-time augmentation).

A view = (engine, image transform). Transforms:
    orig      the image as-is
    flip      mirrored - part of the overlay text is mirrored and unreadable otherwise
    sat       HSV saturation channel - coloured/outlined overlay fonts pop, the photo flattens
    sat_flip  saturation channel, mirrored
Engines: EasyOCR (view name as above) and PaddleOCR PP-OCRv5 mobile (prefix "pm_").

FINAL_VIEWS = EasyOCR orig + Paddle x 4 transforms. The texts of all views are concatenated;
see reports/report.md for the comparison that led to this choice.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_SIDE = 1280  # downscale very large inputs (speed); dataset images are <= 640 px

TRANSFORMS = {
    "orig": lambda im: im,
    "flip": lambda im: im[:, ::-1],
    "sat": lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 1], cv2.COLOR_GRAY2RGB),
    "sat_flip": lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, ::-1, 1], cv2.COLOR_GRAY2RGB),
}
EASYOCR_VIEWS = ("orig", "flip", "sat", "sat_flip")
PADDLE_VIEWS = ("pm_orig", "pm_flip", "pm_sat", "pm_sat_flip")
FINAL_VIEWS = ("orig",) + PADDLE_VIEWS

_ENGINES: dict = {}


def _easyocr():
    if "easyocr" not in _ENGINES:
        import easyocr

        _ENGINES["easyocr"] = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ENGINES["easyocr"]


def _paddle():
    """PaddleOCR mobile models. mkldnn is disabled: on Windows CPU, paddlepaddle 3.x oneDNN
    kernels crash on PP-OCRv5."""
    if "paddle" not in _ENGINES:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        PaddleOCR = _import_paddleocr()
        _ENGINES["paddle"] = PaddleOCR(
            lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=True, enable_mkldnn=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec")
    return _ENGINES["paddle"]


def _import_paddleocr():
    """paddlex optionally imports sklearn.cluster (for table/layout pipelines we don't use).
    Where that compiled module cannot load (e.g. blocked by Windows Application Control),
    hide scikit-learn from paddlex's metadata-based dependency check during the import."""
    try:
        import sklearn.cluster  # noqa: F401
        from paddleocr import PaddleOCR
        return PaddleOCR
    except ImportError:
        pass
    import importlib.metadata as md

    real_version = md.version

    def version(name):
        if name.lower().replace("_", "-") == "scikit-learn":
            raise md.PackageNotFoundError(name)
        return real_version(name)

    md.version = version
    try:
        from paddleocr import PaddleOCR
    finally:
        md.version = real_version
    return PaddleOCR


def _reading_order(items) -> str:
    """items: [(text, polygon)] -> text sorted top-to-bottom (20 px lines), then left-to-right."""
    items = [(t, np.asarray(p, dtype=float)) for t, p in items if str(t).strip()]
    items.sort(key=lambda it: (round(it[1][:, 1].mean() / 20), it[1][:, 0].mean()))
    return " ".join(str(t) for t, _ in items)


def ocr_view(view: str, img: np.ndarray) -> str:
    """OCR one view of an RGB image and return its text."""
    paddle = view.startswith("pm_")
    x = np.ascontiguousarray(TRANSFORMS[view[3:] if paddle else view](img))
    if paddle:
        r = _paddle().predict(np.ascontiguousarray(x[:, :, ::-1]))[0]  # Paddle expects BGR
        return _reading_order(zip(r["rec_texts"], r["rec_polys"]))
    return _reading_order((t, box) for box, t, _ in _easyocr().readtext(x))


def load_image(path: Path) -> np.ndarray:
    """Read an image as RGB; works with non-ASCII paths (cv2.imread does not on Windows)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img


class OCRCache:
    """Per-view OCR text keyed by image content hash, stored as append-only JSONL.
    Reads every file in `read_from`, appends new results to `write_to` (None = memory only)."""

    def __init__(self, read_from=(), write_to: Path | None = None):
        self.data: dict[str, dict[str, str]] = {}
        self.write_to = Path(write_to) if write_to else None
        for f in read_from:
            if Path(f).exists():
                for line in open(f, encoding="utf-8"):
                    rec = json.loads(line)
                    self.data.setdefault(rec["hash"], {}).update(rec["views"])

    def texts(self, path: Path, views=FINAL_VIEWS) -> dict[str, str]:
        """OCR text per view for one image; only missing views are computed."""
        h = hashlib.md5(Path(path).read_bytes()).hexdigest()
        have = self.data.setdefault(h, {})
        missing = [v for v in views if v not in have]
        if missing:
            img = load_image(path)
            new = {v: ocr_view(v, img) for v in missing}
            have.update(new)
            if self.write_to:
                self.write_to.parent.mkdir(parents=True, exist_ok=True)
                with open(self.write_to, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"hash": h, "views": new}, ensure_ascii=False) + "\n")
        return {v: have[v] for v in views}


def ocr_text(path: Path, cache: OCRCache, views=FINAL_VIEWS) -> str:
    """The model input for one image: texts of all views concatenated."""
    return " ".join(t for t in cache.texts(path, views).values() if t)


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
