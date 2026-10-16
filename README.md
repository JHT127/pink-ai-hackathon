# Pink AI 2026 hackathon: tumour measurement classification

This competition exercise cleans a deliberately messy tumour-measurement dataset and predicts malignant (M) versus benign (B), scored by recall at precision >= 0.90. The notebook validates models with patient-grouped cross-validation and keeps test data out of model and threshold selection.

## What we found

- Wrong measurement units could be repaired using radius, perimeter, and area geometry.
- One measurement column had a site-specific offset.
- Patient IDs appeared in four formats; duplicate and conflicting records were handled.
- A post-diagnosis administrative column leaked the label, so it was excluded from model features.
- Some labels may be noisy; a tested noise filter did not meet the adoption rule.

## Results

The final model is plain HistGradientBoosting. Patient-grouped out-of-fold ROC-AUC was 0.9312; at threshold 0.90, precision was 0.939 and recall was 0.560. The threshold is the lowest one meeting the notebook's precision, patient-bootstrap, and lower-prevalence precision floors. At an estimated 22% malignant share, precision was 0.873.

## Repository

- `data/raw/`: competition inputs; `data/clean/`: generated stages.
- `notebooks/main.ipynb`: end-to-end analysis and results.
- `src/`: documented loading, cleaning, repair, audit, and model functions.
- `submissions/`: final prediction file; `logs/`: cleaning and decision records.
- `evidence/`: evidence sheet and figures; `pitch/`: presentation materials.

## How to reproduce

1. From the repository root, create a virtual environment: `python -m venv .venv`.
2. Activate it with `\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux.
3. Install packages: `python -m pip install -r requirements.txt`.
4. Open `notebooks/main.ipynb`, select the environment's Python kernel, and choose **Run All** from a fresh kernel.

The MIT licence covers the code only, not the competition data.

## Limitations and responsible use

This is a competition exercise, not a medical device. It must never be used as a stand-alone diagnosis. Performance may degrade on new centres, scanners, or case mixes.

## AI assistance

AI assistance was used for coding and analysis, with outputs checked against notebook results; details are in [logs/ai_usage_log.md](logs/ai_usage_log.md).
