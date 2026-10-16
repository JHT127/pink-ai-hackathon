# Evidence Sheet

### 1 What we cleaned and why

```
9,000 training labels were mapped to B/M [nb §2]; no unknown labels remained.
11,308 comma-number cells were parsed: 9,919 decimal and 1,389 thousands [nb §2].
Geometry flagged 1,520 train rows for radius/perimeter/area checks before merging [log cleaning #11].
Duplicate resolution merged 396 repeat rows; Site C compactness_se shifted by 0.001 in 1,922 cells (38 set missing) [nb §4] [log cleaning #14].
```

### 2 Problems we found that were not obvious

```
Radius, perimeter, and area obey shared shape geometry; broken ratios exposed scale errors [nb §2].
For each measure, worst should be at least mean; some were reversed [nb §3].
Site C compactness_se showed a +0.001 offset [nb §3].
IDs appear in 4 formats; 50 patients had conflicting duplicate labels [nb §4].
```

### 3 Columns we dropped, and why

```
Drop biopsy_followup_code: grouped AUC 0.977, a diagnosis leak [nb §3].
sample_ref letter/number AUCs: 0.507/0.496; notes presence/language: 0.494/0.499 [nb §3].
center, exam month, and repair flags score about 0.50; patient_id is excluded as an identifier [nb §3].
Drop notes, sample_ref, center, exam_date, patient_id, and repair flags; retain measurements and age [nb §3].
```

### 4 How we validated

```
Validation used 5-fold StratifiedGroupKFold on normalized patient_id; groups never crossed folds [nb §4].
The logistic pipeline fits imputation and scaling within each fold; final HGB handles missing values natively [nb §4].
Leave-one-centre-out checks held out Site A, Site B, and Site C [nb §4].
Final validation uses patient-grouped out-of-fold predictions; no test labels were used [nb §4].
```

### 5 Our threshold, and why

```
Chosen threshold is 0.90: lowest OOF threshold with precision >= 0.93, bootstrap p05 >= 0.93, and 30%-share precision >= 0.91 [nb §5].
At threshold 0.90, OOF precision is 0.939 and recall is 0.560 [nb §5].
Lowering to 0.85 catches 328 more cancers for 45 extra false alarms: 0.14 alarms per extra cancer [nb §5].
At the estimated 22% malignant share, expected precision is 0.873 [nb §5].
```

### 6 What would break this model

```
New centres or scanners, new unit errors, or a changed Site C offset could break calibration [nb §3].
Test malignant share is estimated at 22% versus 39% in training, not a known test label rate [nb §3].
About 4% of training rows are confident OOF mistakes (334 rows); labels may be noisy [nb §5].
This model is triage support, not a stand-alone diagnosis [nb §5].
```

What we used AI assistants for:
We used AI to compare the label-noise filter, report OOF threshold tradeoffs, and draft candidate submissions; we checked metrics and file invariants ourselves [log AI #9].
