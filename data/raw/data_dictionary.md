# Pink AI 2026 - data dictionary

## Where the data comes from

Each record describes a single fine needle aspirate (FNA) of a breast mass. A thin
needle draws a small sample of cells from the mass, the sample is placed on a slide,
and the boundary of each cell nucleus in the image is traced. Ten shape and texture
properties are computed for every nucleus in the slide. Each property is then
summarised across the nuclei in three ways, which is why almost every measurement
appears three times:

- `_mean` - the average over all nuclei on the slide
- `_se` - the standard error of that average, so a measure of how much the nuclei on
  the slide disagree with each other
- `_worst` - the average of the three largest values on the slide

Records in this export were collected at three clinical centres over roughly two
years, and were entered into the registry by hand from several source systems.

## Identifier and context columns

| column | meaning |
| --- | --- |
| `patient_id` | registry identifier for the patient the sample came from |
| `exam_date` | date the sample was taken |
| `center` | clinical centre that produced the slide: Site A, Site B or Site C |
| `age` | patient age in years at the time of the exam |
| `biopsy_followup_code` | administrative code recorded by the registry clerk for this record |
| `notes` | free text typed by whoever entered the record |
| `sample_ref` | internal reference string used by the lab's own slide tracking system |
| `diagnosis` | the target. `M` = malignant, `B` = benign. Training file only. |

## The ten measurements

| measurement | what it describes |
| --- | --- |
| `radius` | mean distance from the centre of a nucleus to points on its perimeter |
| `texture` | standard deviation of the grey-scale values inside the nucleus |
| `perimeter` | length of the traced nuclear boundary |
| `area` | area enclosed by that boundary |
| `smoothness` | local variation in the radius, so how even the boundary is |
| `compactness` | how far the shape departs from a circle, from perimeter and area |
| `concavity` | severity of the inward dents in the boundary |
| `concave_points` | how many inward dents there are, rather than how deep |
| `symmetry` | how closely the two halves of the nucleus match |
| `fractal_dimension` | roughness of the boundary, a "coastline approximation" |

Each appears as `<measurement>_mean`, `<measurement>_se` and `<measurement>_worst`.
Radius, perimeter and area are recorded in millimetres and square millimetres. The
remaining measurements are dimensionless ratios.

Larger, rougher, more irregular nuclei are, broadly, the ones pathologists associate
with malignancy. Beyond that, what matters and how much is for you to work out.

## What you are given

| file | contents |
| --- | --- |
| `pinkai_train.csv` | training records, including `diagnosis` |
| `pinkai_test.csv` | records to predict, with no `diagnosis` column |
| `sample_submission.csv` | the exact submission format, filled with a constant baseline |

## Submission format

Three columns, one row per patient in the test set:

```
patient_id,predicted_label,predicted_prob
910417,B,0.0312
```

- `predicted_label` must be `M` or `B`
- `predicted_prob` is your predicted probability of `M`, a number between 0 and 1
- every test `patient_id` must appear exactly once

Name your file `TEAM_submission_N.csv`, where `TEAM` is your team name and `N` is 1,
2 or 3. You get three scored submissions. `sample_submission.csv` is already a valid
file, so send one early and confirm the scorer accepts it.

## How you are scored

**Recall (sensitivity) at precision >= 0.90**, computed from `predicted_label`.

In plain terms: of all the malignant cases in the test set, what fraction did you
catch - but your score only counts if, among the cases you called malignant, at least
90% really were. A submission that cannot reach 90% precision scores zero.

Ties are broken first by ROC-AUC and then by Brier score, both computed from
`predicted_prob`, so it is worth submitting honest probabilities and not just labels.
