# Pink AI Competition: brief, rules and rubric

Draft v3, 3 hour format, leaderboard bands measured. Everything marked **OPEN** needs a decision before this is published.

## 1. The task

You get a messy medical dataset of breast tumour measurements taken from fine needle aspirate images: missing values, duplicates, wrong units, broken dates, inconsistent labels, junk columns. In 3 hours, working in a team of 4:

1. Clean it.
2. Train a model that separates malignant from benign.
3. Choose a decision threshold you can defend.
4. Submit predictions on the unlabeled test set.
5. Fill a one page evidence sheet and pitch it in 3 minutes.

Nobody is expected to finish everything. The leaderboard and the rubric both give partial credit. Do the parts you can do well.

## 2. What you receive

| File | Contents |
| --- | --- |
| `pinkai_train.csv` | ~9,000 rows, labeled, dirty |
| `pinkai_test.csv` | ~3,000 rows, no label, dirty |
| `sample_submission.csv` | the exact submission format |
| `data_dictionary.md` | what the columns mean |
| `PinkAI_Starter.ipynb` | optional scaffold, use it or ignore it |

The answer key is private.

## 3. Submissions

- Format: `patient_id, predicted_label, predicted_prob`. One row per test patient, no missing values, `predicted_prob` between 0 and 1, `predicted_label` is `M` or `B`.
- Filename: `TEAMNAME_submission.csv`.

**OPEN**: submission channel (shared Drive folder watched by the scorer, or a form).

## 4. Rules

- Teams of 3.
- Python only. `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, and `xgboost` or `lightgbm` if your environment has them. Deep learning earns no extra points.
- **AI assistants are allowed and expected.** Claude Code, Copilot, ChatGPT, anything. Say in the evidence sheet what you used them for. Using them well is a skill we are testing.
- You may not search for the original public dataset this was derived from, and you may not use any external labels.
- Every team answers one judge question about a decision they made. Not being able to explain your own pipeline costs more points than a mediocre score.

## 5. Scoring metric

- **Primary**: recall at precision >= 0.90. Catch as many malignant cases as possible while keeping false alarms under control. Precision below 0.90 scores 0 on this metric.
- **Secondary**: ROC-AUC from `predicted_prob`. Threshold free, so it still rewards a good model even if your threshold is in the wrong place.

Two things to know before you train anything:

1. This is not accuracy, and `predict()` will not optimise it for you.
2. The test set is not drawn the same way as the training set. Clear the 0.90 floor with margin instead of targeting it exactly. Teams that land at 0.89 score zero on the primary metric.

## 6. Evidence sheet (one page)

Six boxes. Three or four lines each. This replaces a written report and should take 20 minutes, not two hours.

1. **What we cleaned and why.** Your top 5 fixes and the rows affected.
2. **Problems we found that were not obvious.** Anything wrong with the *relationships between columns*, not just individual cells.
3. **Columns we dropped, and why.** Name any column you excluded on suspicion and show what made you suspicious.
4. **How we validated.** Your CV scheme, and how you stopped the same patient appearing in train and validation.
5. **Our threshold, and why.** The number, how you picked it, what it costs in false alarms to catch one more cancer.
6. **What would break this model.** Who it would fail, what it must never be used for.

Plus one line: what you used AI assistants for.

## 7. Rubric, 100 points

| Section | Points | What earns it |
| --- | --- | --- |
| Leaderboard: primary metric | 20 | Banded against reference runs, not against the best team |
| Leaderboard: ROC-AUC | 10 | Threshold free, so model and cleaning quality still count |
| Evidence sheet | 30 | Each box scored 0 to 5. Honest and specific beats complete and vague |
| Structural problems found and repaired | 20 | 10 per problem: 5 for finding it, 5 for repairing instead of deleting the rows |
| Pitch and judge question | 20 | 10 for the pitch, 10 for answering one question about your own decisions |

Partial credit is the default. A team that cleans well and never trains a model still scores on cleaning, the evidence sheet and the pitch. There are no all or nothing sections.

Leaderboard bands, measured from `private/benchmarks.md`:

**Primary metric, recall at precision >= 0.90, 20 points**

| Score | Points |
| --- | --- |
| 0.70 and above | 20 |
| 0.66 to 0.699 | 15 to 19 |
| 0.40 to 0.659 | 8 to 14 |
| Above 0 to 0.399 | 4 to 7 |
| 0, precision floor not reached | 0 |

**ROC-AUC, 10 points**

| Score | Points |
| --- | --- |
| 0.973 and above | 10 |
| 0.965 to 0.972 | 8 to 9 |
| 0.900 to 0.964 | 5 to 7 |
| 0.818 to 0.899 | 2 to 4 |
| Below 0.818 | 0 to 1 |

A team can score 0 on the primary metric and still take 10 points here. That is deliberate.

**Deductions**: notebook does not run top to bottom (-10). Test set used to fit anything or to pick the threshold (-20). A number in the evidence sheet that does not appear in the notebook (-10).
