"""Fixed-rule cleaning functions for Stage 3."""

import re
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import LOGS_DIR


MEASUREMENTS = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
)
SUFFIXES = ("mean", "se", "worst")
NUMERIC_COLUMNS = tuple(
    f"{measurement}_{suffix}"
    for measurement in MEASUREMENTS
    for suffix in SUFFIXES
) + ("age",)
TEXT_COLUMNS = (
    "patient_id",
    "exam_date",
    "center",
    "age",
    "biopsy_followup_code",
    "notes",
    "sample_ref",
    "diagnosis",
)
FAKE_MISSING = ("", "nan", "n/a", "na", "?", "null", "none", "missing", "unknown", "-", "--")
DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%b-%y",
    "%d-%B-%y",
    "%B %d, %Y",
    "%b %d, %Y",
)


def _entry(problem, columns, rows_affected, fix, why):
    """Build a consistently shaped cleaning-log record."""
    return {
        "problem": problem,
        "columns": columns,
        "rows_affected": int(rows_affected),
        "fix": fix,
        "why": why,
    }


def rename_columns(df):
    """Normalize headers without changing any cell values."""
    result = df.copy()
    old_columns = list(result.columns)
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in old_columns]
    assert all(re.fullmatch(r"[a-z_]+", column) for column in result.columns)
    changed = sum(old != new for old, new in zip(old_columns, result.columns))
    return result, _entry(
        "Inconsistent column names",
        ", ".join(new for old, new in zip(old_columns, result.columns) if old != new) or "none",
        0,
        f"Renamed {changed} column headers; no data cells changed.",
        "Consistent lowercase names with underscores match the data dictionary.",
    )


def clean_text_columns(df):
    """Strip surrounding whitespace from the designated text columns."""
    result = df.copy()
    changed = 0
    columns = [column for column in TEXT_COLUMNS if column in result.columns]
    for column in columns:
        values = result[column]
        stripped = values.map(lambda value: value.strip() if isinstance(value, str) else value)
        changed += int((values.astype("string") != stripped.astype("string")).fillna(False).sum())
        result[column] = stripped
    return result, _entry(
        "Surrounding whitespace in text fields",
        ", ".join(columns),
        changed,
        f"Stripped surrounding whitespace from {changed} cells (cell count).",
        "Whitespace obscures otherwise identical identifiers, categories, and text values.",
    )


def normalise_patient_id(df):
    """Keep the source identifier and normalize patient IDs to digits only."""
    result = df.copy()
    if "patient_id" not in result.columns:
        return result, _entry(
            "Patient identifiers",
            "patient_id, patient_id_raw",
            0,
            "Skipped because patient_id is absent.",
            "No patient identifier was available to normalize.",
        )
    original = result["patient_id"].copy()
    text = original.astype("string").str.strip()
    text = text.str.replace(r"(?i)^ID-", "", regex=True)
    text = text.str.replace(r"\.0+$", "", regex=True)
    normalized = text.str.replace(r"\D", "", regex=True)
    normalized = normalized.mask(normalized.eq(""), pd.NA)
    changed = int((original.astype("string") != normalized).fillna(False).sum())
    result["patient_id_raw"] = original
    result["patient_id"] = normalized
    return result, _entry(
        "Patient identifiers with prefixes or punctuation",
        "patient_id, patient_id_raw",
        changed,
        f"Preserved the source value in patient_id_raw and normalized {changed} identifiers to digits only.",
        "Duplicate detection and grouped validation must use one canonical patient key.",
    )


def fix_fake_missing(df):
    """Replace configured missing-value spellings in numeric fields with NaN."""
    result = df.copy()
    columns = [column for column in NUMERIC_COLUMNS if column in result.columns]
    token_counts = {token: 0 for token in FAKE_MISSING}
    for column in columns:
        values = result[column]
        normalized = values.map(
            lambda value: value.strip().casefold() if isinstance(value, str) else None
        )
        for token in FAKE_MISSING:
            matches = normalized.eq(token).fillna(False)
            token_counts[token] += int(matches.sum())
            result.loc[matches, column] = np.nan
    changed = sum(token_counts.values())
    token_summary = ", ".join(f"{token!r}: {count}" for token, count in token_counts.items())
    return result, _entry(
        "Fake missing-value text in numeric fields",
        ", ".join(columns),
        changed,
        f"Replaced {changed} cells with NaN; counts by token: {token_summary}.",
        "These fixed missing-value spellings are not numeric observations.",
    )


def parse_numbers(df):
    """Convert numeric fields to float using the fixed comma interpretation rules."""
    result = df.copy()
    columns = [column for column in NUMERIC_COLUMNS if column in result.columns]
    decimal_comma = 0
    thousands_comma = 0
    ambiguous = []
    unparseable = []
    converted_cells = 0

    for column in columns:
        raw_values = result[column]
        parsed_values = []
        for index, raw_value in raw_values.items():
            if pd.isna(raw_value):
                parsed_values.append(np.nan)
                continue

            text = re.sub(r"\s+", "", str(raw_value))
            if "," in text and "." in text:
                thousands_comma += 1
                text = text.replace(",", "")
            elif "," in text:
                decimal_comma += 1
                if re.fullmatch(r"[+-]?\d{1,3},\d{3}", text):
                    ambiguous.append((column, str(raw_value)))
                text = text.replace(",", ".")

            number = pd.to_numeric(text, errors="coerce")
            if pd.isna(number):
                unparseable.append((column, str(raw_value)))
                parsed_values.append(np.nan)
            else:
                parsed_values.append(float(number))
            converted_cells += 1
        result[column] = pd.Series(parsed_values, index=raw_values.index, dtype="float64")

    print(f"Comma numbers: decimal-comma cells={decimal_comma}; thousands-comma cells={thousands_comma}.")
    print(f"Ambiguous comma cells treated as decimal: {len(ambiguous)}; first 10 (column=value): {ambiguous[:10]}")
    print(f"Unparseable numeric cells set to NaN: {len(unparseable)}; remaining non-numeric values (column, value): {unparseable}")
    examples = ", ".join(f"{column}={value!r}" for column, value in ambiguous[:10]) or "none"
    return result, _entry(
        "Numeric fields stored as text",
        ", ".join(columns),
        converted_cells,
        f"Converted {converted_cells} cells to float; decimal-comma={decimal_comma}; thousands-comma={thousands_comma}; unparseable set to NaN={len(unparseable)}; ambiguous treated as decimal={len(ambiguous)}; first 10 examples: {examples}.",
        "The fixed parsing rules normalize number formats without fitting or learning from either file.",
    )


def standardise_labels(df):
    """Normalize known diagnosis spellings and create the integer target."""
    result = df.copy()
    if "diagnosis" not in result.columns:
        return result, _entry(
            "Diagnosis labels",
            "diagnosis, y",
            0,
            "Skipped because diagnosis is absent.",
            "The test file has no target labels to standardize.",
        )

    normalized = result["diagnosis"].astype("string").str.strip().str.lower()
    labels = normalized.map(
        {"b": "B", "benign": "B", "0": "B", "m": "M", "malignant": "M", "1": "M"}
    )
    unknown = normalized.notna() & labels.isna()
    if unknown.any():
        unknown_values = result.loc[unknown, "diagnosis"].value_counts(dropna=False).to_dict()
        print(f"Unrecognized diagnosis values retained as NaN: {unknown_values}")
    result["diagnosis"] = labels
    result["y"] = labels.map({"B": 0, "M": 1}).astype("Int64")

    nan_labels = int(result["diagnosis"].isna().sum())
    assert nan_labels == int(labels.isna().sum())
    class_counts = result["y"].value_counts(dropna=False).sort_index().to_dict()
    malignant_percent = (float((result["y"] == 1).sum()) / len(result) * 100) if len(result) else 0.0
    print(f"NaN diagnosis labels: {nan_labels}")
    print(f"Final class counts (y): {class_counts}; malignant={malignant_percent:.2f}%")
    changed = int((result["diagnosis"].astype("string") != df["diagnosis"].astype("string")).fillna(False).sum())
    changed += int(result["y"].notna().sum())
    return result, _entry(
        "Inconsistent diagnosis labels",
        "diagnosis, y",
        changed,
        f"Standardized known labels to B/M and created integer y; {nan_labels} diagnosis labels are NaN.",
        "Only the specified benign and malignant spellings are mapped; unknown values are not guessed.",
    )


def standardise_center(df):
    """Normalize known centre spellings and reject any other values."""
    result = df.copy()
    if "center" not in result.columns:
        return result, _entry(
            "Centre labels",
            "center",
            0,
            "Skipped because center is absent.",
            "No centre field was available to standardize.",
        )
    normalized = result["center"].map(
        lambda value: value.strip().upper() if isinstance(value, str) else value
    )
    mapping = {"SITE A": "Site A", "SITE B": "Site B", "SITE C": "Site C"}
    result["center"] = normalized.map(mapping)
    invalid = result["center"].isna()
    if invalid.any():
        invalid_values = normalized[invalid].value_counts(dropna=False).to_dict()
        print(f"Unexpected centre values: {invalid_values}")
    print(f"Centre counts: {result['center'].value_counts(dropna=False).to_dict()}")
    assert not invalid.any(), f"Unexpected centre values: {normalized[invalid].unique().tolist()}"
    changed = int((result["center"].astype("string") != df["center"].astype("string")).fillna(False).sum())
    return result, _entry(
        "Inconsistent centre spelling or case",
        "center",
        changed,
        f"Stripped and uppercased values, then standardized {changed} cells to Site A/B/C.",
        "The data dictionary defines exactly three clinical centres.",
    )


def parse_dates(df):
    """Parse exam dates while requiring observed evidence before choosing slash order."""
    result = df.copy()
    if "exam_date" not in result.columns:
        return result, _entry(
            "Exam dates",
            "exam_date, exam_date_raw",
            0,
            "Skipped because exam_date is absent.",
            "No date field was available to parse.",
        )

    raw_dates = result["exam_date"].copy()
    text_dates = raw_dates.map(lambda value: value.strip() if isinstance(value, str) else value)
    slash_mask = text_dates.astype("string").str.contains("/", regex=False, na=False)
    slash_parts = text_dates[slash_mask].astype("string").str.split("/", expand=True)
    if len(slash_parts):
        first = pd.to_numeric(slash_parts[0], errors="coerce")
        second = pd.to_numeric(slash_parts[1], errors="coerce")
        first_over_12 = int((first > 12).sum())
        second_over_12 = int((second > 12).sum())
        fully_ambiguous = int(((first <= 12) & (second <= 12)).sum())
    else:
        first_over_12 = second_over_12 = fully_ambiguous = 0
    print(
        "Slash-date ambiguity: "
        f"first part > 12={first_over_12}; second part > 12={second_over_12}; "
        f"both <= 12={fully_ambiguous}."
    )
    if slash_mask.any() and not (first_over_12 > 0 and second_over_12 == 0):
        raise ValueError(
            "Cannot choose slash-date order from observed values; "
            f"first part > 12={first_over_12}, second part > 12={second_over_12}."
        )

    parsed = []
    for value in text_dates:
        if pd.isna(value) or not str(value).strip():
            parsed.append(pd.NaT)
            continue
        text = str(value).strip()
        date_value = pd.NaT
        for date_format in DATE_FORMATS:
            try:
                date_value = pd.Timestamp(datetime.strptime(text, date_format))
                break
            except ValueError:
                continue
        parsed.append(date_value)
    result["exam_date_raw"] = raw_dates
    result["exam_date"] = pd.to_datetime(parsed)

    unparseable = int(result["exam_date"].isna().sum())
    minimum = result["exam_date"].min()
    maximum = result["exam_date"].max()
    lower = pd.Timestamp("2024-01-01")
    upper = pd.Timestamp("2026-12-31")
    outside = result["exam_date"].notna() & (
        (result["exam_date"] < lower) | (result["exam_date"] > upper)
    )
    print(f"Unparseable dates set to NaT: {unparseable}")
    print(f"Date range: min={minimum}; max={maximum}; outside 2024-01-01..2026-12-31={int(outside.sum())}")
    changed = int(result["exam_date"].notna().sum()) + int(result["exam_date_raw"].notna().sum())
    return result, _entry(
        "Exam dates stored in mixed formats",
        "exam_date, exam_date_raw",
        changed,
        f"Parsed valid dates, retained original text in exam_date_raw, and set {unparseable} unparseable dates to NaT; slash counts: first part > 12={first_over_12}, second part > 12={second_over_12}, both <= 12={fully_ambiguous}; range={minimum}..{maximum}; outside allowed range={int(outside.sum())}.",
        "Explicit formats and measured slash-date evidence avoid guessing; dates outside the expected window are reported, not changed.",
    )


def clean_stage3(df):
    """Apply the fixed Stage 3 cleaning rules and return the transformed frame and log."""
    result = df.copy()
    entries = []
    for cleaning_function in (
        rename_columns,
        normalise_patient_id,
        clean_text_columns,
        fix_fake_missing,
        parse_numbers,
        standardise_labels,
        standardise_center,
        parse_dates,
    ):
        result, log_entry = cleaning_function(result)
        entries.append(log_entry)
    return result, entries


def write_cleaning_log(entries):
    """Replace only the cleaning-log table, numbering entries from one."""
    path = LOGS_DIR / "cleaning_log.md"
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("| #"))
    separator_index = header_index + 1
    table_end = separator_index + 1
    while table_end < len(lines) and lines[table_end].startswith("|"):
        table_end += 1
    table_rows = []
    for number, entry in enumerate(entries, start=1):
        values = [
            str(number),
            str(entry["problem"]),
            str(entry["columns"]),
            str(entry["rows_affected"]),
            str(entry["fix"]),
            str(entry["why"]),
        ]
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        table_rows.append("| " + " | ".join(escaped) + " |")
    lines[header_index + 2 : table_end] = table_rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")