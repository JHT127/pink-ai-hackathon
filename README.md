# Pink AI 2026

Clean the breast-tumour dataset and predict malignant (M) vs benign (B), selecting a threshold for recall at precision >= 0.90.

## Folder tree

```text
.github/
	copilot-instructions.md
data/
	raw/                    # Provided data; never edit
	clean/
notebooks/
	main.ipynb
src/
	__init__.py
	config.py
submissions/
logs/
	cleaning_log.md
	decisions_log.md
	ai_usage_log.md
evidence/
	evidence_sheet.md
pitch/
```

## How to run

1. Create a virtual environment: `python -m venv .venv`
2. Activate it, then install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/main.ipynb` and select **Run All**.

Do not edit the raw data files in `data/raw/`.
