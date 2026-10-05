"""OCR the labelled dataset (images/alcohol, images/non_alcohol) into a table for training.

    python -m src.run_ocr --workers 4             # all 8 views (final 5 + EasyOCR views for the comparison)
    python -m src.run_ocr --views final           # only the 5 views the final model needs

Results are cached per view in outputs/ocr_cache*.jsonl, so the run is resumable and re-runs
are instant. Output: outputs/ocr_dataset.csv with one `text_<view>` column per view.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .ocr import EASYOCR_VIEWS, FINAL_VIEWS, OCRCache, list_images

LABELS = {"alcohol": 1, "non_alcohol": 0}
OUT = Path("outputs")


def dataset(data: Path) -> list[tuple[Path, int]]:
    return [(p, y) for sub, y in LABELS.items() for p in list_images(data / sub)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="images")
    ap.add_argument("--views", default="all", help="'all', 'final' or a comma-separated list")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=-1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    views = {"all": EASYOCR_VIEWS + FINAL_VIEWS[1:], "final": FINAL_VIEWS}.get(args.views) or tuple(args.views.split(","))
    items = dataset(Path(args.data))
    caches = sorted(OUT.glob("ocr_cache*.jsonl"))

    if args.shard >= 0:  # worker process: OCR every n-th image into its own cache file
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // args.workers)))
        cache = OCRCache(caches, OUT / f"ocr_cache_{args.shard}.jsonl")
        for p, _ in tqdm(items[args.shard::args.workers], desc=f"worker {args.shard}", position=args.shard):
            cache.texts(p, views)
        return

    if args.workers > 1:
        procs = [subprocess.Popen([sys.executable, "-m", "src.run_ocr", *sys.argv[1:], "--shard", str(k)])
                 for k in range(args.workers)]
        if any(p.wait() for p in procs):
            sys.exit("an OCR worker failed")
        caches = sorted(OUT.glob("ocr_cache*.jsonl"))

    cache = OCRCache(caches, OUT / "ocr_cache.jsonl")
    rows = []
    for p, y in tqdm(items, desc="table"):
        texts = cache.texts(p, views)
        rows.append({"file_name": p.name, "path": p.as_posix(), "label": y, **{f"text_{v}": t for v, t in texts.items()}})
    pd.DataFrame(rows).to_csv(OUT / "ocr_dataset.csv", index=False, encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {OUT / 'ocr_dataset.csv'}")


if __name__ == "__main__":
    main()
