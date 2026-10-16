"""Stage 6 model evaluation, threshold selection, and submission helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES_DROP, ID_COL, RANDOM_SEED, SAMPLE_SUBMISSION, SUBMISSIONS_DIR
from src.repair import MEASUREMENT_COLUMNS


DEFAULT_VARIANT = "base"
MODEL_VARIANTS = ("base", "winsorize", "ratios", "winsorize+ratios")


class Winsorizer(TransformerMixin, BaseEstimator):
    """Clip features to fold-fitted 0.5th and 99.5th percentiles."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(values, 0.005, axis=0)
        self.upper_ = np.nanquantile(values, 0.995, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)


class RatioFeatureAdder(TransformerMixin, BaseEstimator):
    """Add deterministic measurement ratios while retaining feature names."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.feature_names_in_)
        return add_ratio_features(frame)


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


def add_ratio_features(df):
    """Return a copy with the four requested ratios; zero denominators become NaN."""
    result = df.copy()
    ratios = {
        "perimeter_mean_radius_mean_ratio": ("perimeter_mean", "radius_mean"),
        "area_mean_radius_mean_squared_ratio": ("area_mean", "radius_mean", 2),
        "radius_worst_radius_mean_ratio": ("radius_worst", "radius_mean"),
        "concavity_worst_concavity_mean_ratio": ("concavity_worst", "concavity_mean"),
    }
    for feature, terms in ratios.items():
        numerator = pd.to_numeric(result[terms[0]], errors="coerce")
        denominator = pd.to_numeric(result[terms[1]], errors="coerce")
        if len(terms) == 3:
            denominator = denominator.pow(terms[2])
        result[feature] = numerator.div(denominator.where(denominator.ne(0)))
    return result


def _model_pipeline(classifier, variant, scale=False):
    steps = []
    if "ratios" in variant:
        steps.append(("ratio_features", RatioFeatureAdder()))
    if "winsorize" in variant:
        steps.append(("winsorizer", Winsorizer()))
    if scale:
        steps.extend(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def make_models(variant=None):
    """Create fresh model templates using the requested preprocessing variant."""
    selected_variant = DEFAULT_VARIANT if variant is None else variant
    if selected_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {selected_variant}")
    logreg = _model_pipeline(
        LogisticRegression(C=1.0, max_iter=2000), selected_variant, scale=True
    )
    hgb = _model_pipeline(
        HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        random_state=RANDOM_SEED,
        ),
        selected_variant,
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


def fit_with_noise_filter(X, y, groups, model_factory, threshold=0.10):
    """Fit a model after removing rows with low inner-OOF probability for their label.

    Returns the fitted estimator, a Boolean keep mask, and each row's inner-OOF
    probability for its observed label. Filtering is confined to these inputs.
    """
    target = np.asarray(y, dtype=int)
    group_values = np.asarray(groups)
    if len(X) != len(target) or len(target) != len(group_values):
        raise ValueError("X, y, and groups must have equal lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    own_label_probability = np.full(len(target), np.nan, dtype=float)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    for train_index, validation_index in splitter.split(X, target, group_values):
        train_groups = set(group_values[train_index])
        validation_groups = set(group_values[validation_index])
        assert train_groups.isdisjoint(validation_groups)
        fitted = clone(model_factory()).fit(X.iloc[train_index], target[train_index])
        class_index = int(np.flatnonzero(fitted.classes_ == 1)[0])
        probability = fitted.predict_proba(X.iloc[validation_index])[:, class_index]
        own_label_probability[validation_index] = np.where(
            target[validation_index] == 1, probability, 1 - probability
        )

    if not np.isfinite(own_label_probability).all():
        raise ValueError("Inner grouped OOF filtering did not score every training row")
    keep_mask = own_label_probability >= threshold
    if len(np.unique(target[keep_mask])) < 2:
        raise ValueError("Noise filter retained fewer than two label classes")
    final_model = clone(model_factory()).fit(X.iloc[keep_mask], target[keep_mask])
    return final_model, keep_mask, own_label_probability


def compare_noise_filter(X, y, groups, model_factory, threshold=0.10):
    """Compare plain and filtered models on identical outer patient-grouped OOF folds."""
    target = np.asarray(y, dtype=int)
    group_values = np.asarray(groups)
    if len(X) != len(target) or len(target) != len(group_values):
        raise ValueError("X, y, and groups must have equal lengths")
    predictions = {
        "plain": np.full(len(target), np.nan, dtype=float),
        "filtered": np.full(len(target), np.nan, dtype=float),
    }
    removed_rows = 0
    training_rows = 0
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, validation_index in splitter.split(X, target, group_values):
        train_groups = set(group_values[train_index])
        validation_groups = set(group_values[validation_index])
        assert train_groups.isdisjoint(validation_groups)

        plain_model = clone(model_factory()).fit(X.iloc[train_index], target[train_index])
        class_index = int(np.flatnonzero(plain_model.classes_ == 1)[0])
        predictions["plain"][validation_index] = plain_model.predict_proba(
            X.iloc[validation_index]
        )[:, class_index]

        filtered_model, keep_mask, _ = fit_with_noise_filter(
            X.iloc[train_index],
            target[train_index],
            group_values[train_index],
            model_factory,
            threshold=threshold,
        )
        class_index = int(np.flatnonzero(filtered_model.classes_ == 1)[0])
        predictions["filtered"][validation_index] = filtered_model.predict_proba(
            X.iloc[validation_index]
        )[:, class_index]
        removed_rows += int((~keep_mask).sum())
        training_rows += len(train_index)

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Outer grouped OOF evaluation did not predict every row")
    removed_share = removed_rows / training_rows if training_rows else 0.0
    rows = []
    for name, probability in predictions.items():
        rows.append(
            {
                "variant": name,
                **metrics(target, probability),
                "share_training_rows_removed": removed_share if name == "filtered" else 0.0,
            }
        )
    return pd.DataFrame(rows), predictions


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


def measurement_outlier_report(df):
    """Summarize six-IQR outlier shares and five most extreme values per measure."""
    rows = []
    for column in MEASUREMENT_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        observed = values.dropna()
        median = float(observed.median())
        lower_quartile = float(observed.quantile(0.25))
        upper_quartile = float(observed.quantile(0.75))
        spread = upper_quartile - lower_quartile
        outside = observed.lt(median - 6 * spread) | observed.gt(median + 6 * spread)
        extreme_values = (
            observed.loc[(observed - median).abs().sort_values(ascending=False).index]
            .head(5)
            .tolist()
        )
        rows.append(
            {
                "column": column,
                "n": int(len(observed)),
                "beyond_6_iqr": int(outside.sum()),
                "share_beyond_6_iqr": float(outside.mean()) if len(observed) else np.nan,
                "five_most_extreme_values": extreme_values,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["share_beyond_6_iqr", "beyond_6_iqr"], ascending=False
    ).reset_index(drop=True)


def compare_model_variants(X, y, groups):
    """Evaluate logistic regression and HGB on identical patient-grouped OOF folds."""
    prediction_map = {}
    metric_rows = []
    for variant in MODEL_VARIANTS:
        predictions = oof_predictions(
            {name: model for name, model in make_models(variant).items() if name in {"hgb", "logreg"}},
            X,
            y,
            groups,
        )
        for name, probability in predictions.items():
            prediction_map[(variant, name)] = probability
            metric_rows.append(
                {"variant": variant, "model": name, **metrics(y, probability)}
            )
    return pd.DataFrame(metric_rows), prediction_map


def best_model_variant(comparison, minimum_auc_gain=0.003):
    """Return the highest-AUC HGB variant meeting the minimum gain over base."""
    hgb = comparison.loc[comparison["model"].eq("hgb")].set_index("variant")
    base_auc = float(hgb.loc["base", "roc_auc"])
    eligible = hgb.loc[hgb["roc_auc"].ge(base_auc + minimum_auc_gain)]
    if eligible.empty:
        return "base"
    return str(eligible["roc_auc"].idxmax())


def threshold_options(y, probability, groups, n_bootstrap=200):
    """Return natural and prevalence-weighted threshold metrics with patient bootstrap p05."""
    target = np.asarray(y, dtype=int)
    predicted_probability = np.asarray(probability, dtype=float)
    group_values = pd.Series(groups).astype("string").fillna("__missing__")
    if len(target) != len(predicted_probability) or len(target) != len(group_values):
        raise ValueError("y, probability, and groups must have equal lengths")
    prevalence = float(target.mean())
    group_codes, unique_groups = pd.factorize(group_values, sort=False)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_weights = np.zeros((n_bootstrap, len(unique_groups)), dtype=np.int16)
    for sample_index in range(n_bootstrap):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        bootstrap_weights[sample_index] = np.bincount(
            sampled, minlength=len(unique_groups)
        )

    rows = []
    for threshold in np.round(np.arange(0.50, 1.00, 0.01), 2):
        predicted = predicted_probability >= threshold
        true_positive = predicted & (target == 1)
        false_positive = predicted & (target == 0)
        tp = int(true_positive.sum())
        fp = int(false_positive.sum())
        total_predicted = tp + fp
        natural_precision = tp / total_predicted if total_predicted else 0.0
        recall = tp / int((target == 1).sum()) if (target == 1).any() else 0.0
        group_tp = np.bincount(group_codes, weights=true_positive, minlength=len(unique_groups))
        group_fp = np.bincount(group_codes, weights=false_positive, minlength=len(unique_groups))
        bootstrap_tp = bootstrap_weights @ group_tp
        bootstrap_fp = bootstrap_weights @ group_fp
        bootstrap_precision = np.divide(
            bootstrap_tp,
            bootstrap_tp + bootstrap_fp,
            out=np.zeros(n_bootstrap, dtype=float),
            where=(bootstrap_tp + bootstrap_fp) > 0,
        )
        row = {
            "threshold": float(threshold),
            "precision_39": natural_precision,
            "recall": recall,
            "bootstrap_p05_39": float(np.percentile(bootstrap_precision, 5)),
            "TP": tp,
            "FP": fp,
        }
        for share in (0.30, 0.22, 0.15):
            benign_weight = prevalence * (1 - share) / (share * (1 - prevalence))
            weighted_fp = fp * benign_weight
            row[f"precision_{int(share * 100)}"] = (
                tp / (tp + weighted_fp) if tp + weighted_fp else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def highest_recall_thresholds(table, precision_floor=0.92):
    """Find the highest-recall threshold meeting a precision floor at each share."""
    rows = []
    for share, column in (
        (0.39, "precision_39"),
        (0.30, "precision_30"),
        (0.22, "precision_22"),
        (0.15, "precision_15"),
    ):
        eligible = table.loc[table[column].ge(precision_floor)]
        if eligible.empty:
            rows.append(
                {"malignant_share": share, "threshold": np.nan, "precision": np.nan, "recall": np.nan}
            )
            continue
        selected = eligible.loc[eligible["recall"].idxmax()]
        rows.append(
            {
                "malignant_share": share,
                "threshold": float(selected["threshold"]),
                "precision": float(selected[column]),
                "recall": float(selected["recall"]),
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_options(table, output_path):
    """Save the four prevalence-specific precision curves and recall curve."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for share, column in (
        (39, "precision_39"),
        (30, "precision_30"),
        (22, "precision_22"),
        (15, "precision_15"),
    ):
        axis.plot(table["threshold"], table[column], label=f"Precision, M share {share}%")
    axis.plot(table["threshold"], table["recall"], color="black", linestyle="--", label="Recall")
    axis.set(xlabel="Threshold", ylabel="Score", ylim=(0, 1), title="OOF precision and recall options")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


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


def write_candidate_submissions(test_df, probability, thresholds=(0.85, 0.90), team_name="TEAM"):
    """Write validated threshold candidates from one shared test probability vector."""
    if ID_COL not in test_df.columns:
        raise ValueError(f"Submission input is missing {ID_COL}")
    probability = np.asarray(probability, dtype=float)
    if len(probability) != len(test_df):
        raise ValueError("Test rows and probabilities must have equal lengths")
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Submission probabilities must be finite and in [0, 1]")

    sample = pd.read_csv(SAMPLE_SUBMISSION, dtype={ID_COL: "string"})
    if len(sample) != 2884 or len(test_df) != 2884:
        raise ValueError("Candidate submissions require exactly 2,884 test and sample rows")
    test_keys = test_df[ID_COL].map(_id_key)
    sample_keys = sample[ID_COL].map(_id_key)
    if test_keys.duplicated().any() or sample_keys.duplicated().any():
        raise ValueError("Test and sample submission patient_ids must each be unique")
    if set(test_keys) != set(sample_keys):
        raise ValueError("Test patient_ids do not match the sample submission")

    probability_by_key = pd.Series(probability, index=test_keys)
    if probability_by_key.isna().any():
        raise ValueError("Candidate probabilities could not be aligned to sample patient_ids")
    aligned_probability = probability_by_key.reindex(sample_keys).to_numpy()
    output_dir = Path(SUBMISSIONS_DIR) / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for threshold in thresholds:
        label = f"t{int(round(threshold * 100)):03d}"
        submission = pd.DataFrame(
            {
                ID_COL: sample[ID_COL],
                "predicted_label": np.where(aligned_probability >= threshold, "M", "B"),
                "predicted_prob": aligned_probability,
            }
        )
        if list(submission.columns) != [ID_COL, "predicted_label", "predicted_prob"]:
            raise ValueError("Submission must contain exactly the three required columns")
        if submission.isna().any().any():
            raise ValueError("Submission contains missing values")
        if not submission["predicted_label"].isin(["M", "B"]).all():
            raise ValueError("Submission labels must be M or B")
        if not submission["predicted_prob"].between(0, 1).all():
            raise ValueError("Submission probabilities must be in [0, 1]")

        output_path = output_dir / f"{team_name}_{label}.csv"
        submission.to_csv(output_path, index=False, float_format="%.6f")
        saved = pd.read_csv(output_path, dtype={ID_COL: "string"})
        if len(saved) != 2884 or saved.isna().any().any():
            raise ValueError("Written candidate failed row-count or missing-value validation")
        if not saved[ID_COL].equals(sample[ID_COL]):
            raise ValueError("Written candidate patient_ids are not in sample order")
        if not saved["predicted_label"].isin(["M", "B"]).all():
            raise ValueError("Written candidate labels must be M or B")
        if not saved["predicted_prob"].between(0, 1).all():
            raise ValueError("Written candidate probabilities must be in [0, 1]")
        outputs[threshold] = submission
        print(f"{output_path.name}: test predicted M share={(submission['predicted_label'] == 'M').mean():.4f}")
    return outputs