# Pink AI 2026 Copilot Instructions

- Goal: clean the breast-tumour data and predict malignant (M) vs benign (B), optimizing recall at precision >= 0.90.
- Layout: `data/`, `notebooks/`, `src/`, `submissions/`, `logs/`, `evidence/`, and `pitch/`.
- Raw CSVs have a junk banner on row 1; the real header is row 2. Read them with `header=1`.
- Never edit raw data; write cleaned data to `data/clean/`.
- NEVER fit, tune, scale, impute, or choose a threshold using the test file. Learn all statistics (including medians and scalers) from train only, then apply them to test. Use test only for final prediction.
- Repair rows rather than deleting them wherever possible.
- Group cross-validation by `patient_id`; a patient must never appear in both train and validation. Use random seed 42 everywhere.
- Log every cleaning step in `logs/cleaning_log.md`: problem, columns, rows affected, fix, and why.
- Put logic in documented functions in `src/*.py`. The notebook only imports and calls them and prints every number that may be quoted in the evidence sheet.
- The notebook must run top to bottom without errors.
- Write simple, readable code with short comments. I must be able to explain every line to a judge.
- Do not create extra files or folders unless asked.
- Keep answers short; do not re-explain code unless asked.
