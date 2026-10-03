"""Read-only summaries for inspecting the raw project data."""

import re

import pandas as pd

from src.config import ID_COL, TARGET


def shape_summary(train, test):
    """Return file shapes and columns present in only one file."""
    # Compare names without renaming either input.
    train_columns = set(train.columns)
    test_columns = set(test.columns)
    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "train_only_columns": sorted(train_columns - test_columns),
        "test_only_columns": sorted(test_columns - train_columns),
    }


def odd_column_names(df):
    """Return column names that are not lowercase letters and underscores."""
    # Keep the original names visible in the returned table.
    return pd.DataFrame(
        {"column_name": [name for name in df.columns if not re.fullmatch(r"[a-z_]+", name)]}
    )


def label_counts(train):
    """Return training diagnosis counts using their raw spellings."""
    # Do not normalize labels while reporting them.
    counts = train[TARGET].value_counts(dropna=False)
    return counts.rename_axis("raw_value").reset_index(name="count")


def empty_cell_counts(df):
    """Return empty-like cell counts and percentages for every column."""
    # Treat blank text and the literal text nan as empty for this report.
    rows = []
    for column in df.columns:
        values = df[column].astype(str)
        empty = values.str.strip().eq("") | values.str.strip().str.lower().eq("nan")
        count = int(empty.sum())
        rows.append(
            {
                "column": column,
                "empty_count": count,
                "empty_percent": (count / len(df) * 100) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["empty_count", "column"], ascending=[False, True], ignore_index=True
    )


def comma_decimal_count(df, columns):
    """Count raw cells matching a comma-decimal numeric pattern per column."""
    # Match the requested shape without converting source values.
    pattern = re.compile(r"^-?\d+,\d+$")
    rows = []
    for column in columns:
        values = df[column].astype(str)
        rows.append(
            {"column": column, "comma_decimal_count": int(values.map(lambda value: bool(pattern.fullmatch(value.strip()))).sum())}
        )
    return pd.DataFrame(rows)


def non_numeric_cells(df, columns):
    """Count unconvertible non-empty values and show up to five examples."""
    # Replace decimal commas only in temporary values used for this report.
    rows = []
    for column in columns:
        raw_values = df[column].astype(str)
        non_empty = raw_values.str.strip().ne("") & raw_values.str.strip().str.lower().ne("nan")
        converted = pd.to_numeric(raw_values.str.replace(",", ".", regex=False), errors="coerce")
        bad = non_empty & converted.isna()
        rows.append(
            {
                "column": column,
                "non_numeric_count": int(bad.sum()),
                "example_bad_values": raw_values[bad].drop_duplicates().head(5).tolist(),
            }
        )
    return pd.DataFrame(rows)


def id_report(train, test):
    """Return duplicate, unique-ID, exact-row, and shared-ID summaries."""
    # Compare the raw ID strings from both files.
    train_ids = train[ID_COL]
    test_ids = test[ID_COL]
    shared_ids = sorted(set(train_ids) & set(test_ids))
    return {
        "train_duplicate_patient_id_rows": int(train_ids.duplicated().sum()),
        "test_duplicate_patient_id_rows": int(test_ids.duplicated().sum()),
        "train_identical_duplicate_rows": int(train.duplicated().sum()),
        "test_identical_duplicate_rows": int(test.duplicated().sum()),
        "train_unique_patient_ids": int(train_ids.nunique(dropna=False)),
        "test_unique_patient_ids": int(test_ids.nunique(dropna=False)),
        "shared_patient_ids": shared_ids,
        "shared_patient_id_count": len(shared_ids),
    }


def category_values(df, col):
    """Return counts of raw values for one categorical column."""
    # Include blank strings and preserve each raw category spelling.
    counts = df[col].value_counts(dropna=False)
    return counts.rename_axis("raw_value").reset_index(name="count")


def date_samples(df, n=15):
    """Return a deterministic random sample of raw exam_date values."""
    # Use the project seed so the sample is repeatable.
    return df[["exam_date"]].sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)


def numeric_ranges(df, columns):
    """Return numeric min, percentiles, median, and max for selected columns."""
    # Convert temporary series only; leave the input DataFrame untouched.
    rows = []
    for column in columns:
        values = pd.to_numeric(
            df[column].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        ).dropna()
        rows.append(
            {
                "column": column,
                "min": values.min(),
                "p01": values.quantile(0.01),
                "median": values.median(),
                "p99": values.quantile(0.99),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def measurement_columns(df):
    """Return measurement column names ending in mean, se, or worst."""
    # Follow the requested suffix rule while preserving original column names.
    suffixes = ("_mean", "_se", "_worst")
    columns = [
        column
        for column in df.columns
        if column.lower().replace(" ", "_").endswith(suffixes)
    ]
    return pd.DataFrame({"column": columns})