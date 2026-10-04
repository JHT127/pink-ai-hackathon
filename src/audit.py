"""Read-only Stage 5 audits for suspicious columns and measurement signal."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.repair import MEASUREMENT_COLUMNS


def _notes_text(df):
    """Return normalized notes strings without changing the input frame."""
    return df["notes"].astype("string").fillna("").str.strip()


def _sample_ref_number(df):
    """Extract all digits from sample_ref as a numeric value."""
    digits = df["sample_ref"].astype("string").str.replace(r"\D", "", regex=True)
    return pd.to_numeric(digits, errors="coerce")


def _sample_ref_letter(df):
    """Return the final non-missing character of sample_ref."""
    return df["sample_ref"].astype("string").str.strip().str[-1]


def _candidate_features(train):
    """Build the requested single-column candidates in a separate frame."""
    notes = _notes_text(train)
    language = pd.Series("other", index=train.index, dtype="string")
    language.loc[notes.eq("")] = "empty"
    language.loc[notes.str.contains(r"[\u0600-\u06ff]", regex=True)] = "Arabic"
    language.loc[
        notes.str.contains(r"[A-Za-z]", regex=True) & ~language.eq("Arabic")
    ] = "English"

    features = {
        "biopsy_followup_code": train["biopsy_followup_code"],
        "center": train["center"],
        "age": pd.to_numeric(train["age"], errors="coerce"),
        "exam_date_month": pd.to_datetime(train["exam_date"], errors="coerce").dt.month,
        "sample_ref_number": _sample_ref_number(train),
        "sample_ref_letter": _sample_ref_letter(train),
        "notes_present": notes.ne(""),
        "notes_language": language,
    }
    for column in train.columns:
        if column.startswith("flag_"):
            features[column] = train[column].astype("string").fillna("(missing)")
    measurement_columns = [column for column in MEASUREMENT_COLUMNS if column in train]
    features["n_missing"] = train[measurement_columns].isna().sum(axis=1)
    return features


def _grouped_auc(values, target, groups, categorical):
    """Return mean and standard deviation of five patient-grouped AUCs."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fit_index, validation_index in splitter.split(values, target, groups):
        if categorical:
            model = make_pipeline(
                OneHotEncoder(handle_unknown="ignore"),
                LogisticRegression(max_iter=1000, random_state=42),
            )
            fit_values = values.iloc[fit_index].astype("string").fillna("(missing)").to_frame()
            validation_values = values.iloc[validation_index].astype("string").fillna("(missing)").to_frame()
        else:
            model = HistGradientBoostingClassifier(
                random_state=42, early_stopping=False
            )
            fit_values = pd.to_numeric(values.iloc[fit_index], errors="coerce").to_frame()
            validation_values = pd.to_numeric(values.iloc[validation_index], errors="coerce").to_frame()
        model.fit(fit_values, target.iloc[fit_index])
        probabilities = model.predict_proba(validation_values)[:, 1]
        scores.append(roc_auc_score(target.iloc[validation_index], probabilities))
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def single_column_power(train):
    """Return grouped cross-validated ROC-AUC for each requested feature alone."""
    target = train["y"].astype(int)
    groups = train["patient_id"]
    categorical = {
        "biopsy_followup_code",
        "center",
        "sample_ref_letter",
        "notes_present",
        "notes_language",
    }
    categorical.update(column for column in train if column.startswith("flag_"))
    rows = []
    for name, values in _candidate_features(train).items():
        mean_auc, std_auc = _grouped_auc(
            values, target, groups, categorical=name in categorical
        )
        rows.append({"feature": name, "auc": mean_auc, "auc_std": std_auc})
    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)


def followup_vs_label(train):
    """Return biopsy follow-up counts and within-code diagnosis percentages."""
    codes = train["biopsy_followup_code"].astype("string").fillna("(missing)")
    counts = pd.crosstab(codes, train["y"]).reindex(columns=[0, 1], fill_value=0)
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    result = counts.rename(columns={0: "benign_count", 1: "malignant_count"})
    result["count"] = counts.sum(axis=1)
    result["benign_pct"] = counts[0].div(row_totals).mul(100)
    result["malignant_pct"] = counts[1].div(row_totals).mul(100)
    return result.reset_index(names="biopsy_followup_code")


def notes_vs_label(train):
    """Return the 15 common non-empty notes and the empty-notes malignant rate."""
    notes = _notes_text(train)
    malignant = train["y"].eq(1)
    rows = []
    nonempty = notes.ne("")
    frequent = notes.loc[nonempty].value_counts().head(15).index
    for text in frequent:
        mask = notes.eq(text)
        rows.append(
            {
                "notes_text": text,
                "count": int(mask.sum()),
                "malignant_pct": float(malignant.loc[mask].mean() * 100),
            }
        )
    empty = ~nonempty
    rows.append(
        {
            "notes_text": "(empty)",
            "count": int(empty.sum()),
            "malignant_pct": float(malignant.loc[empty].mean() * 100) if empty.any() else np.nan,
        }
    )
    return pd.DataFrame(rows)


def sample_ref_vs_label(train):
    """Return malignant rates by sample_ref ending and numeric-reference correlation."""
    letters = _sample_ref_letter(train).fillna("(missing)")
    malignant = train["y"].eq(1)
    rows = []
    for letter in letters.unique():
        mask = letters.eq(letter)
        rows.append(
            {
                "item": "sample_ref_letter",
                "value": letter,
                "count": int(mask.sum()),
                "malignant_pct": float(malignant.loc[mask].mean() * 100),
                "correlation": np.nan,
            }
        )
    reference_number = _sample_ref_number(train)
    valid = reference_number.notna() & train["y"].notna()
    correlation = reference_number.loc[valid].corr(train.loc[valid, "y"])
    rows.append(
        {
            "item": "sample_ref_number_vs_y",
            "value": "all",
            "count": int(valid.sum()),
            "malignant_pct": np.nan,
            "correlation": correlation,
        }
    )
    return pd.DataFrame(rows)


def se_geometry_check(train):
    """Return standard-error medians by unit-repair flag, including geometry ratios."""
    radius = pd.to_numeric(train["radius_se"], errors="coerce")
    perimeter = pd.to_numeric(train["perimeter_se"], errors="coerce")
    area = pd.to_numeric(train["area_se"], errors="coerce")
    valid_radius = radius.gt(0)
    area_ratio = area.div(radius.pow(2)).where(valid_radius)
    perimeter_ratio = perimeter.div(radius).where(valid_radius)
    values = pd.DataFrame(
        {
            "radius_se": radius,
            "perimeter_se": perimeter,
            "area_se": area,
            "area_se/radius_se^2": area_ratio,
            "perimeter_se/radius_se": perimeter_ratio,
        },
        index=train.index,
    )
    return values.groupby(train["flag_unit_fixed"].fillna(False).astype(bool)).median().rename_axis(
        "flag_unit_fixed"
    ).reset_index()


def train_test_shift(train, test):
    """Return largest train-IQR-scaled median shifts and descriptive category shares."""
    numeric_columns = [column for column in MEASUREMENT_COLUMNS if column in train and column in test]
    if "age" in train and "age" in test:
        numeric_columns.append("age")
    rows = []
    for column in numeric_columns:
        train_values = pd.to_numeric(train[column], errors="coerce")
        test_values = pd.to_numeric(test[column], errors="coerce")
        median_train = train_values.median()
        median_test = test_values.median()
        iqr_train = train_values.quantile(0.75) - train_values.quantile(0.25)
        rows.append(
            {
                "metric": "median",
                "column": column,
                "category": "",
                "median_train": median_train,
                "median_test": median_test,
                "shift_iqr_train": (median_test - median_train) / iqr_train if iqr_train else np.nan,
                "percent_train": np.nan,
                "percent_test": np.nan,
            }
        )
    measurement_rows = pd.DataFrame(rows)
    measurement_rows = measurement_rows.reindex(
        measurement_rows["shift_iqr_train"].abs().sort_values(ascending=False).head(10).index
    )

    share_rows = []
    for column in ("biopsy_followup_code", "center"):
        train_values = train[column].astype("string").fillna("(missing)")
        test_values = test[column].astype("string").fillna("(missing)")
        categories = sorted(set(train_values.unique()) | set(test_values.unique()))
        for category in categories:
            share_rows.append(
                {
                    "metric": "percent",
                    "column": column,
                    "category": category,
                    "median_train": np.nan,
                    "median_test": np.nan,
                    "shift_iqr_train": np.nan,
                    "percent_train": float(train_values.eq(category).mean() * 100),
                    "percent_test": float(test_values.eq(category).mean() * 100),
                }
            )
    if "flag_unit_fixed" in train and "flag_unit_fixed" in test:
        share_rows.append(
            {
                "metric": "percent_true",
                "column": "flag_unit_fixed",
                "category": "True",
                "median_train": np.nan,
                "median_test": np.nan,
                "shift_iqr_train": np.nan,
                "percent_train": float(train["flag_unit_fixed"].fillna(False).astype(bool).mean() * 100),
                "percent_test": float(test["flag_unit_fixed"].fillna(False).astype(bool).mean() * 100),
            }
        )
    return pd.concat([measurement_rows, pd.DataFrame(share_rows)], ignore_index=True)


def measurement_signal(train):
    """Return direction-adjusted in-sample ROC-AUC for each measurement column."""
    target = train["y"].astype(int)
    rows = []
    for column in MEASUREMENT_COLUMNS:
        values = pd.to_numeric(train[column], errors="coerce")
        valid = values.notna() & target.notna()
        auc = roc_auc_score(target.loc[valid], values.loc[valid])
        rows.append(
            {
                "column": column,
                "auc": max(float(auc), 1.0 - float(auc)),
                "rows": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)