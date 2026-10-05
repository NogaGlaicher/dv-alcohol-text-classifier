"""Classify every image in a directory and write results.csv.

    python predict.py --input_dir path/to/images [--output results.csv] [--cache ocr_cache.jsonl]

For each image:
  1. OCR: EasyOCR on the original image + PaddleOCR on 4 views (original, mirrored,
     saturation channel, mirrored saturation); the texts are concatenated (src/ocr.py).
  2. The trained model (models/model.joblib) cleans the text, scores it with a TF-IDF
     logistic regression fused with the alcohol lexicon, and returns p(alcohol) (src/features.py).
  3. p(alcohol) >= the threshold saved with the model -> "alcohol", otherwise "non_alcohol".

Output: a CSV with columns file_name,prediction, in the format of example_results.csv.
Images are searched recursively (jpg/jpeg/png/bmp/webp/tif); unreadable files are logged and
predicted non_alcohol. Runs on CPU at about 10 s/image; the first run downloads the OCR weights.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.ocr import OCRCache, list_images, ocr_text  # noqa: E402

log = logging.getLogger("predict")


def main() -> None:
    """Parse arguments, OCR and classify all images, write the results CSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="directory with the images to classify")
    ap.add_argument("--output", default="results.csv", help="output CSV path (default: results.csv)")
    ap.add_argument("--cache", default=None,
                    help="optional JSONL file that stores OCR results, so an interrupted run resumes "
                         "and re-runs are instant")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bundle = joblib.load(ROOT / "models" / "model.joblib")
    paths = list_images(Path(args.input_dir))
    if not paths:
        sys.exit(f"no images found in {args.input_dir}")
    log.info("%d images; OCR on CPU takes ~10 s/image", len(paths))

    cache = OCRCache([args.cache] if args.cache else [], args.cache)
    texts = []
    for p in tqdm(paths):
        try:
            texts.append(ocr_text(p, cache, bundle["views"]))
        except Exception as e:  # corrupt or unsupported file
            log.warning("OCR failed for %s: %s", p, e)
            texts.append("")

    prob = bundle["model"].predict_proba(pd.DataFrame({"text": texts}))[:, 1]
    pred = ["alcohol" if pr >= bundle["threshold"] else "non_alcohol" for pr in prob]
    results = pd.DataFrame({"file_name": [p.name for p in paths], "prediction": pred})
    results.to_csv(args.output, index=False, lineterminator="\n")  # same format as example_results.csv
    log.info("wrote %s (%d alcohol / %d non_alcohol)", args.output, pred.count("alcohol"), pred.count("non_alcohol"))


if __name__ == "__main__":
    main()
