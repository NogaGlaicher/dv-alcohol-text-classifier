# Alcohol-text image classifier (OCR + NLP)

Detects images whose superimposed text references alcoholic beverages, on CPU only.
OCR with two engines on several image views → text cleaning → TF-IDF logistic regression fused with an alcohol lexicon.

|                        | Precision |Recall |   F1  | ROC-AUC |
|------------------------|-----------|-------|-------|---------|
| 5-fold CV (880 images) | 0.985     | 0.914 | 0.948 | 0.993   |
| Hold-out (220 images)  | 0.981     | 0.927 | 0.953 | 0.996   |

Report: [reports/report.md](reports/report.md) · Notebook: [notebooks/01_eda_and_results.ipynb](notebooks/01_eda_and_results.ipynb)

## Setup (Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps easyocr==1.7.2   # avoids a second OpenCV package clashing with PaddleOCR's
```

## Inference

```bash
python predict.py --input_dir path/to/images --output results.csv
```

* Writes `file_name,prediction` with `alcohol` / `non_alcohol`. Images are searched recursively.
* Speed: 8-13 s/image on a 4-core laptop CPU. The first run downloads the OCR weights (~150 MB).
* `--cache ocr_cache.jsonl` saves OCR results, so interrupted runs resume and re-runs are instant.

## Reproducing the results

```bash
python -m src.run_ocr --workers 4   # OCR of images/ -> outputs/ocr_dataset.csv  (~2 h 45 min; cached in outputs/ocr_cache.jsonl)
python -m src.train                 # CV, comparisons, hold-out, final model      (~4 min)
python -m src.evaluate              # OCR metric, unseen-brand test, figures       (~2.5 min)
```

The OCR cache is included, so the first command finishes instantly and the other two reproduce `outputs/metrics.json`
exactly. Re-running OCR from scratch with other library versions can change a few characters of EasyOCR output; on a
20-image check the lexicon decisions were unchanged.

## Layout

```
predict.py             inference script -> results.csv
src/ocr.py             OCR engines, image views, cache
src/text_normalize.py  lowercase, de-leet, de-duplicate, OCR-confusion fixes
src/lexicon.py         alcohol terms, decoys, negation, template words, fuzzy matching
src/features.py        TF-IDF logistic regression + lexicon late fusion
src/run_ocr.py         OCR of the labelled dataset (parallel, resumable)
src/train.py           model selection, comparisons, hold-out, final model
src/evaluate.py        OCR metric, unseen-brand test, figures
models/model.joblib    trained model + threshold
outputs/               OCR table and cache, metrics.json, predictions.csv
```

Windows notes: PaddleOCR runs with `enable_mkldnn=False`, because its oneDNN kernels crash on Windows CPU. Create the venv on
a short path, because deep Jupyter paths break `pip install` otherwise.
