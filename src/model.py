"""Stage 6 model evaluation, threshold selection, and submission helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES_DROP, ID_COL, RANDOM_SEED, SAMPLE_SUBMISSION, SUBMISSIONS_DIR
from src.repair import MEASUREMENT_COLUMNS


class BlendClassifier(ClassifierMixin, BaseEstimator):
    """Average probabilities from logistic regression and histogram boosting."""

    def __init__(self, logreg, hgb):
        self.logreg = logreg
        self.hgb = hgb

    def fit(self, X, y):
        self.logreg_ = clone(self.logreg).fit(X, y)
        self.hgb_ = clone(self.hgb).fit(X, y)
        self.classes_ = self.logreg_.classes_
        return self

    def predict_proba(self, X):
        logreg_probability = self.logreg_.predict_proba(X)
        hgb_probability = self.hgb_.predict_proba(X)
        return (logreg_probability + hgb_probability) / 2


def get_features(df):
    """Return the 30 measurements and age, excluding identifiers and leakage fields."""
    features = list(MEASUREMENT_COLUMNS) + ["age"]
    excluded = set(FEATURES_DROP) | {
        ID_COL,
        "center",
        "notes",
        "sample_ref",
        "biopsy_followup_code",
        "y",
        "diagnosis",
    }
    excluded.update(column for column in df.columns if column.startswith("flag_"))
    excluded.update(column for column in df.columns if column.startswith("exam_date"))
    assert not (set(features) & excluded)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Required model features are missing: {missing}")
    return df.loc[:, features].apply(pd.to_numeric, errors="coerce")


def make_models():
    """Create fresh model templates for fold-wise fitting."""
    logreg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )
    hgb = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        random_state=RANDOM_SEED,
    )
    return {"logreg": logreg, "hgb": hgb, "blend": BlendClassifier(logreg, hgb)}


def oof_predictions(models, X, y, groups):
    """Return patient-grouped, out-of-fold malignant probabilities per model."""
    target = np.asarray(y, dtype=int)
    group_values = np.asarray(groups)
    predictions = {name: np.full(len(X), np.nan) for name in models}
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    )
    for train_index, validation_index in splitter.split(X, target, group_values):
        train_groups = set(group_values[train_index])
        validation_groups = set(group_values[validation_index])
        assert train_groups.isdisjoint(validation_groups)
        for name, template in models.items():
            fitted = clone(template).fit(X.iloc[train_index], target[train_index])
            class_index = int(np.flatnonzero(fitted.classes_ == 1)[0])
            predictions[name][validation_index] = fitted.predict_proba(
                X.iloc[validation_index]
            )[:, class_index]
    for values in predictions.values():
        assert np.isfinite(values).all()
    return predictions


def metrics(y, p):
    """Calculate ranking, calibration, and recall at 90% precision."""
    target = np.asarray(y, dtype=int)
    probability = np.asarray(p, dtype=float)
    positives = target == 1
    thresholds = np.unique(probability)
    eligible = []
    for threshold in thresholds:
        predicted = probability >= threshold
        true_positives = int(np.sum(predicted & positives))
        false_positives = int(np.sum(predicted & ~positives))
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
        if precision >= 0.90:
            recall = true_positives / int(positives.sum()) if positives.any() else 0.0
            eligible.append((recall, float(threshold)))
    recall, threshold = max(eligible, key=lambda item: (item[0], -item[1])) if eligible else (0.0, np.nan)
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "recall_at_precision": float(recall),
        "threshold_at_precision": float(threshold),
    }


def center_holdout(models, df, X, y):
    """Fit on two centers and report metrics for each held-out center."""
    centers = df["center"].astype("string").to_numpy()
    target = np.asarray(y, dtype=int)
    rows = []
    for center in pd.unique(centers):
        validation_mask = centers == center
        train_index = np.flatnonzero(~validation_mask)
        validation_index = np.flatnonzero(validation_mask)
        for name, template in models.items():
            fitted = clone(template).fit(X.iloc[train_index], target[train_index])
            class_index = int(np.flatnonzero(fitted.classes_ == 1)[0])
            probability = fitted.predict_proba(X.iloc[validation_index])[:, class_index]
            rows.append(
                {"model": name, "held_out_center": str(center), **metrics(target[validation_index], probability)}
            )
    return pd.DataFrame(rows)


def choose_threshold(y, p, groups):
    """Select the lowest robust threshold using OOF predictions only."""
    target = np.asarray(y, dtype=int)
    probability = np.asarray(p, dtype=float)
    group_values = np.asarray(groups)
    if len(target) != len(probability) or len(target) != len(group_values):
        raise ValueError("y, p, and groups must have equal lengths")
    prevalence = float(target.mean())
    benign_weight = prevalence * 0.80 / (0.20 * (1 - prevalence))
    sample_weight = np.where(target == 0, benign_weight, 1.0)
    unique_groups, group_index = np.unique(group_values, return_inverse=True)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_group_weights = np.zeros((200, len(unique_groups)), dtype=np.int16)
    for sample_index in range(200):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        bootstrap_group_weights[sample_index] = np.bincount(
            sampled, minlength=len(unique_groups)
        )

    rows = []
    for threshold in np.round(np.arange(0.30, 0.951, 0.01), 2):
        predicted = probability >= threshold
        true_positive = predicted & (target == 1)
        false_positive = predicted & (target == 0)
        tp = int(true_positive.sum())
        fp = int(false_positive.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        weighted_tp = float(sample_weight[true_positive].sum())
        weighted_fp = float(sample_weight[false_positive].sum())
        shifted_precision = weighted_tp / (weighted_tp + weighted_fp) if weighted_tp + weighted_fp else 0.0
        group_tp = np.bincount(group_index, weights=true_positive, minlength=len(unique_groups))
        group_fp = np.bincount(group_index, weights=false_positive, minlength=len(unique_groups))
        bootstrap_tp = bootstrap_group_weights @ group_tp
        bootstrap_fp = bootstrap_group_weights @ group_fp
        bootstrap_precision = np.divide(
            bootstrap_tp,
            bootstrap_tp + bootstrap_fp,
            out=np.zeros(200, dtype=float),
            where=(bootstrap_tp + bootstrap_fp) > 0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": tp / int((target == 1).sum()),
                "shifted_precision": shifted_precision,
                "bootstrap_p05": float(np.percentile(bootstrap_precision, 5)),
                "TP": tp,
                "FP": fp,
            }
        )
    table = pd.DataFrame(rows)

    conditions = [
        (0.94, 0.90, 0.92, "None"),
        (0.94, 0.90, 0.90, "bootstrap p05 >= 0.90"),
        (0.94, 0.88, 0.90, "bootstrap p05 >= 0.90; shifted precision >= 0.88"),
    ]
    eligible = pd.DataFrame()
    relaxed_condition = "None"
    for original_floor, shifted_floor, bootstrap_floor, description in conditions:
        eligible = table.loc[
            table["precision"].ge(original_floor)
            & table["shifted_precision"].ge(shifted_floor)
            & table["bootstrap_p05"].ge(bootstrap_floor)
        ]
        if not eligible.empty:
            relaxed_condition = description
            break
    if eligible.empty:
        raise ValueError("No threshold met the specified constraints after both relaxations")
    chosen_row = eligible.iloc[0]
    chosen_threshold = float(chosen_row["threshold"])
    plain_floor = table.loc[table["precision"].ge(0.90)]
    if plain_floor.empty:
        false_alarms_per_extra_cancer = np.nan
    else:
        lower_row = plain_floor.iloc[0]
        extra_cancers = int(lower_row["TP"] - chosen_row["TP"])
        extra_false_alarms = int(lower_row["FP"] - chosen_row["FP"])
        false_alarms_per_extra_cancer = (
            extra_false_alarms / extra_cancers if extra_cancers > 0 else np.nan
        )
    return table, chosen_threshold, relaxed_condition, float(false_alarms_per_extra_cancer)


def _id_key(value):
    """Normalize integer-like CSV identifiers for stable matching."""
    text = str(value).strip()
    if text.upper().startswith("ID-"):
        text = text[3:]
    if "," in text and text.replace(",", "").isdigit():
        text = text.replace(",", "")
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def make_submission(model, X_train, y_train, test_df, threshold, team_name):
    """Fit on all training rows, validate, and write the ordered submission."""
    if ID_COL not in test_df.columns:
        raise ValueError(f"Submission input is missing {ID_COL}")
    fitted = clone(model).fit(X_train, np.asarray(y_train, dtype=int))
    probability = fitted.predict_proba(get_features(test_df))[:, int(np.flatnonzero(fitted.classes_ == 1)[0])]
    predicted_label = np.where(probability >= threshold, "M", "B")
    sample = pd.read_csv(SAMPLE_SUBMISSION, dtype={ID_COL: "string"})
    test_keys = test_df[ID_COL].map(_id_key)
    sample_keys = sample[ID_COL].map(_id_key)
    if test_keys.duplicated().any() or sample_keys.duplicated().any():
        raise ValueError("Test and sample submission patient_ids must each be unique")
    if set(test_keys) != set(sample_keys):
        test_only = sorted(set(test_keys) - set(sample_keys))[:5]
        sample_only = sorted(set(sample_keys) - set(test_keys))[:5]
        raise ValueError(
            f"Test patient_ids do not match the sample submission; "
            f"test-only={test_only}, sample-only={sample_only}"
        )
    predictions = pd.DataFrame(
        {"_key": test_keys, "predicted_label": predicted_label, "predicted_prob": probability}
    ).set_index("_key")
    submission = pd.DataFrame({ID_COL: sample[ID_COL]})
    submission["predicted_label"] = sample_keys.map(predictions["predicted_label"]).to_numpy()
    submission["predicted_prob"] = sample_keys.map(predictions["predicted_prob"]).to_numpy()
    if list(submission.columns) != [ID_COL, "predicted_label", "predicted_prob"]:
        raise ValueError("Submission must contain exactly the three required columns")
    if len(submission) != len(sample):
        raise ValueError("Submission row count does not match the sample submission")
    if submission[ID_COL].isna().any() or submission[ID_COL].map(_id_key).duplicated().any():
        raise ValueError("Submission patient_ids must be present exactly once")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not submission["predicted_label"].isin(["M", "B"]).all():
        raise ValueError("Submission labels must be M or B")
    if not submission["predicted_prob"].between(0, 1).all():
        raise ValueError("Submission probabilities must be in [0, 1]")

    output_path = Path(SUBMISSIONS_DIR) / f"{team_name}_submission_1.csv"
    submission.to_csv(output_path, index=False, float_format="%.6f")
    saved = pd.read_csv(output_path, dtype={ID_COL: "string"})
    if len(saved) != len(sample) or saved.isna().any().any():
        raise ValueError("Written submission failed row-count or missing-value validation")
    if set(saved[ID_COL].map(_id_key)) != set(sample_keys) or saved[ID_COL].map(_id_key).duplicated().any():
        raise ValueError("Written submission patient_ids failed validation")
    if not saved["predicted_label"].isin(["M", "B"]).all() or not saved["predicted_prob"].between(0, 1).all():
        raise ValueError("Written submission labels or probabilities failed validation")
    print(f"Submission checks passed: {len(saved)} rows, 3 columns, IDs unique and aligned, no NaN.")
    print(f"Test predicted M share: {(predicted_label == 'M').mean():.4f}")
    return submission