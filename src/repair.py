"""Stage 4 structural diagnostics and conservative data repairs."""

import re

import numpy as np
import pandas as pd


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
MEASUREMENT_COLUMNS = tuple(
    f"{measurement}_{suffix}"
    for measurement in MEASUREMENTS
    for suffix in SUFFIXES
)
GEOMETRY = {
    "perimeter_radius": (2 * np.pi, 0.1),
    "area_radius_squared": (np.pi, 0.1),
}
AMBIGUOUS_COMMA = re.compile(r"\d{1,3},\d{3}")


def _entry(problem, columns, rows_affected, fix, why):
    return {
        "problem": problem,
        "columns": columns,
        "rows_affected": int(rows_affected),
        "fix": fix,
        "why": why,
    }


def _normal_ratio(values, typical):
    values = pd.to_numeric(values, errors="coerce")
    valid = values.notna() & np.isfinite(values) & (values > 0)
    normal = pd.Series(False, index=values.index)
    normal.loc[valid] = np.abs(np.log10(values.loc[valid] / typical)) <= 0.1
    return normal


def _ratios(df, suffix):
    radius = pd.to_numeric(df[f"radius_{suffix}"], errors="coerce")
    perimeter = pd.to_numeric(df[f"perimeter_{suffix}"], errors="coerce")
    area = pd.to_numeric(df[f"area_{suffix}"], errors="coerce")
    valid_pr = radius.gt(0) & perimeter.gt(0) & radius.notna() & perimeter.notna()
    valid_ar = radius.gt(0) & area.gt(0) & radius.notna() & area.notna()
    pr = pd.Series(np.nan, index=df.index, dtype="float64")
    ar = pd.Series(np.nan, index=df.index, dtype="float64")
    pr.loc[valid_pr] = perimeter.loc[valid_pr] / radius.loc[valid_pr]
    ar.loc[valid_ar] = area.loc[valid_ar] / radius.loc[valid_ar] ** 2
    return radius, perimeter, area, pr, ar


def _geometry_category(radius, pr, ar):
    pr_ok = bool(_normal_ratio(pd.Series([pr]), GEOMETRY["perimeter_radius"][0]).iloc[0]) if pd.notna(pr) else False
    ar_ok = bool(_normal_ratio(pd.Series([ar]), GEOMETRY["area_radius_squared"][0]).iloc[0]) if pd.notna(ar) else False
    if pr_ok and ar_ok:
        if pd.notna(radius) and not 3 <= radius <= 40:
            return "all_three_scaled"
        return "normal"
    if pd.notna(pr) and pd.notna(ar):
        d_pr = int(round(np.log10(pr / GEOMETRY["perimeter_radius"][0])))
        d_ar = int(round(np.log10(ar / GEOMETRY["area_radius_squared"][0])))
        if d_pr and d_ar == 2 * d_pr:
            return "radius_off"
        if d_pr and d_ar == 0:
            return "perimeter_off"
        if d_pr == 0 and d_ar:
            return "area_off"
    if pr_ok and pd.notna(ar):
        return "area_off"
    if ar_ok and pd.notna(pr):
        return "radius_off_or_perimeter_off"
    return "other_inconsistent"


def diagnose_geometry(df):
    """Return ratio distributions, geometry counts, negatives, and ordering checks."""
    distribution_rows = []
    count_rows = []
    stratified_rows = []
    negative_rows = []
    order_rows = []

    for suffix in SUFFIXES:
        radius, perimeter, area, pr, ar = _ratios(df, suffix)
        for ratio_name, ratio, typical in (
            ("perimeter/radius", pr, GEOMETRY["perimeter_radius"][0]),
            ("area/radius^2", ar, GEOMETRY["area_radius_squared"][0]),
        ):
            valid = ratio.notna() & np.isfinite(ratio) & ratio.gt(0)
            deviations = np.log10(ratio.loc[valid] / typical)
            rounded = deviations.round().astype(int)
            counts = rounded.value_counts().sort_index()
            for exponent, count in counts.items():
                distribution_rows.append(
                    {
                        "suffix": suffix,
                        "ratio": ratio_name,
                        "rounded_log10_scale": int(exponent),
                        "rows": int(count),
                        "log10_min": float(deviations[rounded == exponent].min()),
                        "log10_median": float(deviations[rounded == exponent].median()),
                        "log10_max": float(deviations[rounded == exponent].max()),
                    }
                )

        categories = pd.Series(
            [
                _geometry_category(radius.iloc[pos], pr.iloc[pos], ar.iloc[pos])
                if pd.notna(radius.iloc[pos]) and pd.notna(pr.iloc[pos]) and pd.notna(ar.iloc[pos])
                else "incomplete"
                for pos in range(len(df))
            ],
            index=df.index,
        )
        for category, count in categories.value_counts().sort_index().items():
            count_rows.append({"suffix": suffix, "category": category, "rows": int(count)})
        for group_name, group_values in (
            ("center", df.get("center", pd.Series("unknown", index=df.index))),
            ("month", pd.to_datetime(df.get("exam_date", pd.Series(pd.NaT, index=df.index)), errors="coerce").dt.to_period("M").astype("string").fillna("NaT")),
        ):
            grouped = pd.DataFrame({"group": group_values.astype("string").fillna("unknown"), "category": categories})
            totals = grouped.groupby(["group", "category"], dropna=False).size()
            for (group, category), count in totals.items():
                stratified_rows.append(
                    {"suffix": suffix, "by": group_name, "group": str(group), "category": category, "rows": int(count)}
                )

        for column in (f"{name}_{suffix}" for name in MEASUREMENTS):
            values = pd.to_numeric(df[column], errors="coerce")
            negative_rows.append({"column": column, "negative_values": int(values.lt(0).sum())})
        for left, right, check in (
            (f"radius_{suffix}", f"perimeter_{suffix}", "radius > perimeter"),
            (f"perimeter_{suffix}", f"area_{suffix}", "perimeter > area"),
            (f"{suffix}", f"{suffix}", "worst < mean"),
        ):
            if check == "worst < mean":
                if suffix == "worst":
                    for measure in ("radius", "perimeter", "area"):
                        mask = pd.to_numeric(df[f"{measure}_worst"], errors="coerce") < pd.to_numeric(df[f"{measure}_mean"], errors="coerce")
                        order_rows.append({"check": f"{measure}_worst < {measure}_mean", "rows": int(mask.sum())})
            else:
                mask = pd.to_numeric(df[left], errors="coerce") > pd.to_numeric(df[right], errors="coerce")
                order_rows.append({"check": f"{check} ({suffix})", "rows": int(mask.sum())})

    return {
        "ratio_distribution": pd.DataFrame(distribution_rows),
        "geometry_counts": pd.DataFrame(count_rows),
        "stratified_counts": pd.DataFrame(stratified_rows),
        "negative_counts": pd.DataFrame(negative_rows),
        "ordering_counts": pd.DataFrame(order_rows),
    }


def fix_negative_values(df):
    """Repair negative geometry values only when absolute values restore normal ratios."""
    result = df.copy()
    result["flag_sign_fixed"] = False
    outcomes = {}
    changed_rows = set()
    for suffix in SUFFIXES:
        for measure in MEASUREMENTS:
            column = f"{measure}_{suffix}"
            if column not in result:
                continue
            values = pd.to_numeric(result[column], errors="coerce")
            for index in values.index[values.lt(0)]:
                candidate = abs(float(values.loc[index]))
                if measure not in ("radius", "perimeter", "area"):
                    consistent = False
                else:
                    row = result.loc[[index]].copy()
                    row[column] = candidate
                    _, _, _, pr, ar = _ratios(row, suffix)
                    checks = []
                    if measure in ("radius", "perimeter") and pd.notna(pr.iloc[0]):
                        checks.append(bool(_normal_ratio(pr, GEOMETRY["perimeter_radius"][0]).iloc[0]))
                    if measure in ("radius", "area") and pd.notna(ar.iloc[0]):
                        checks.append(bool(_normal_ratio(ar, GEOMETRY["area_radius_squared"][0]).iloc[0]))
                    consistent = bool(checks) and all(checks)
                outcome = "absolute_value" if consistent else "set_nan"
                outcomes.setdefault(column, {"absolute_value": 0, "set_nan": 0})[outcome] += 1
                if consistent:
                    result.at[index, column] = candidate
                    result.at[index, "flag_sign_fixed"] = True
                else:
                    result.at[index, column] = np.nan
                changed_rows.add(index)
    detail = "; ".join(
        f"{column}: abs={counts['absolute_value']}, NaN={counts['set_nan']}"
        for column, counts in sorted(outcomes.items())
    ) or "no negative measurements"
    fixed = sum(counts["absolute_value"] for counts in outcomes.values())
    missing = sum(counts["set_nan"] for counts in outcomes.values())
    return result, _entry(
        "Negative measurement values",
        ", ".join(outcomes) or "measurement columns",
        len(changed_rows),
        f"Absolute-value repairs={fixed}; set to NaN={missing}; per column: {detail}.",
        "Absolute values were kept only when the available geometry ratios returned to the observed normal range.",
    )


def _normalized_raw_columns(raw_df):
    return {
        str(column).strip().lower().replace(" ", "_"): column
        for column in raw_df.columns
    }


def _percentile_reference(reference, excluded=None):
    ranges = {}
    excluded = excluded or {}
    for column in MEASUREMENT_COLUMNS:
        if column not in reference:
            continue
        values = pd.to_numeric(reference[column], errors="coerce")
        if column in excluded:
            values = values.drop(index=excluded[column], errors="ignore")
        values = values.dropna()
        if len(values):
            ranges[column] = (float(values.quantile(0.01)), float(values.quantile(0.99)))
    return ranges


def _candidate_geometry_ok(row, column, candidate):
    measure, suffix = column.rsplit("_", 1)
    temporary = row.copy()
    temporary[column] = candidate
    _, _, _, pr, ar = _ratios(temporary, suffix)
    if measure == "radius":
        checks = [
            bool(_normal_ratio(pr, GEOMETRY["perimeter_radius"][0]).iloc[0]) if pd.notna(pr.iloc[0]) else None,
            bool(_normal_ratio(ar, GEOMETRY["area_radius_squared"][0]).iloc[0]) if pd.notna(ar.iloc[0]) else None,
        ]
    elif measure == "perimeter":
        checks = [bool(_normal_ratio(pr, GEOMETRY["perimeter_radius"][0]).iloc[0]) if pd.notna(pr.iloc[0]) else None]
    elif measure == "area":
        checks = [bool(_normal_ratio(ar, GEOMETRY["area_radius_squared"][0]).iloc[0]) if pd.notna(ar.iloc[0]) else None]
    else:
        return None
    available = [check for check in checks if check is not None]
    return bool(available) and all(available)


def resolve_ambiguous_commas(df_stage3, raw_df, train_reference=None):
    """Resolve raw comma-only three-decimal-digit values using geometry or train ranges."""
    result = df_stage3.copy()
    result["flag_comma_resolved"] = False
    raw_columns = _normalized_raw_columns(raw_df)
    references = train_reference if train_reference is not None else df_stage3
    excluded = {}
    if train_reference is None:
        for normalized, original in raw_columns.items():
            if normalized not in result or normalized not in MEASUREMENT_COLUMNS:
                continue
            hits = raw_df[original].astype(str).str.strip().map(lambda value: bool(AMBIGUOUS_COMMA.fullmatch(value)))
            excluded[normalized] = raw_df.index[hits]
    ranges = _percentile_reference(references, excluded if train_reference is None else None)
    outcomes = {}
    touched_rows = set()
    for column in MEASUREMENT_COLUMNS:
        raw_column = raw_columns.get(column)
        if raw_column is None or column not in result:
            continue
        matches = raw_df[raw_column].astype(str).str.strip().map(lambda value: bool(AMBIGUOUS_COMMA.fullmatch(value)))
        for index in raw_df.index[matches]:
            if index not in result.index:
                continue
            text = str(raw_df.at[index, raw_column]).strip()
            decimal = float(text.replace(",", "."))
            thousands = float(text.replace(",", ""))
            candidates = {"kept_decimal": decimal, "switched_to_thousands": thousands}
            plausible = []
            if column.startswith(("radius_", "perimeter_", "area_")):
                row = result.loc[[index]]
                plausible = [
                    outcome
                    for outcome, value in candidates.items()
                    if _candidate_geometry_ok(row, column, value)
                ]
            else:
                low_high = ranges.get(column)
                if low_high:
                    plausible = [
                        outcome
                        for outcome, value in candidates.items()
                        if low_high[0] <= value <= low_high[1]
                    ]
            outcome = plausible[0] if len(plausible) == 1 else "set_nan"
            result.at[index, column] = candidates[plausible[0]] if len(plausible) == 1 else np.nan
            result.at[index, "flag_comma_resolved"] = True
            touched_rows.add(index)
            column_outcomes = outcomes.setdefault(
                column,
                {"kept_decimal": 0, "switched_to_thousands": 0, "set_nan": 0},
            )
            column_outcomes[outcome] += 1
    totals = {
        key: sum(value[key] for value in outcomes.values())
        for key in ("kept_decimal", "switched_to_thousands", "set_nan")
    }
    detail = "; ".join(
        f"{column}: decimal={values['kept_decimal']}, thousands={values['switched_to_thousands']}, NaN={values['set_nan']}"
        for column, values in sorted(outcomes.items())
    ) or "no ambiguous comma cells"
    return result, _entry(
        "Ambiguous comma-only numeric values",
        ", ".join(outcomes) or "measurement columns",
        len(touched_rows),
        f"Kept decimal={totals['kept_decimal']}; switched to thousands={totals['switched_to_thousands']}; set to NaN={totals['set_nan']}; per column: {detail}.",
        "Raw text was re-matched; geometry was used for radius/perimeter/area and the training 1st-99th percentile range for other measurements.",
    )


def _apply_scale(values, column, suffix, factor):
    candidate = values.copy()
    candidate[column] = candidate[column] / factor
    return candidate


def _unit_candidates(row, suffix):
    radius, perimeter, area, pr, ar = _ratios(row, suffix)
    r, p, a = radius.iloc[0], perimeter.iloc[0], area.iloc[0]
    current_pr, current_ar = pr.iloc[0], ar.iloc[0]
    candidates = []
    if pd.notna(current_pr) and pd.notna(current_ar) and _normal_ratio(pr, GEOMETRY["perimeter_radius"][0]).iloc[0] and _normal_ratio(ar, GEOMETRY["area_radius_squared"][0]).iloc[0] and pd.notna(r) and not 3 <= r <= 40:
        exponent = int(round(np.log10(r / 20)))
        exponent = max(1, exponent) if r > 40 else min(-1, exponent)
        factor = 10.0 ** exponent
        corrected_r = r / factor
        if 3 <= corrected_r <= 40:
            corrected = row.copy()
            corrected[f"radius_{suffix}"] = r / factor
            corrected[f"perimeter_{suffix}"] = p / factor
            corrected[f"area_{suffix}"] = a / (factor ** 2)
            _, _, _, corrected_pr, corrected_ar = _ratios(corrected, suffix)
            if _normal_ratio(corrected_pr, GEOMETRY["perimeter_radius"][0]).iloc[0] and _normal_ratio(corrected_ar, GEOMETRY["area_radius_squared"][0]).iloc[0]:
                candidates.append(("all_three_scaled", None, factor))
    current_normal = (
        pd.notna(current_pr)
        and pd.notna(current_ar)
        and bool(_normal_ratio(pr, GEOMETRY["perimeter_radius"][0]).iloc[0])
        and bool(_normal_ratio(ar, GEOMETRY["area_radius_squared"][0]).iloc[0])
    )
    if not current_normal and (pd.notna(current_pr) or pd.notna(current_ar)):
        d_pr = int(round(np.log10(current_pr / GEOMETRY["perimeter_radius"][0]))) if pd.notna(current_pr) else None
        d_ar = int(round(np.log10(current_ar / GEOMETRY["area_radius_squared"][0]))) if pd.notna(current_ar) else None
        inferred = []
        if d_pr is not None and d_ar is not None:
            if d_pr and d_ar == 2 * d_pr:
                inferred.append(("radius", -d_pr))
            if d_pr and d_ar == 0:
                inferred.append(("perimeter", d_pr))
            if d_pr == 0 and d_ar:
                inferred.append(("area", d_ar))
        elif d_pr is not None and d_pr:
            if not 3 <= r <= 40:
                inferred.append(("radius", -d_pr))
            else:
                inferred.append(("perimeter", d_pr))
        elif d_ar is not None and d_ar:
            if not 3 <= r <= 40:
                if d_ar % 2 == 0:
                    inferred.append(("radius", -(d_ar // 2)))
            else:
                inferred.append(("area", d_ar))
        for measure, exponent in inferred:
            if exponent == 0 or abs(exponent) > 6:
                continue
            column = f"{measure}_{suffix}"
            factor = 10.0 ** exponent
            corrected = _apply_scale(row, column, suffix, factor)
            _, _, _, corrected_pr, corrected_ar = _ratios(corrected, suffix)
            pr_pass = pd.isna(current_pr) or bool(_normal_ratio(corrected_pr, GEOMETRY["perimeter_radius"][0]).iloc[0])
            ar_pass = pd.isna(current_ar) or bool(_normal_ratio(corrected_ar, GEOMETRY["area_radius_squared"][0]).iloc[0])
            corrected_radius = pd.to_numeric(corrected[f"radius_{suffix}"], errors="coerce").iloc[0]
            physical_pass = measure != "radius" or 3 <= corrected_radius <= 40
            if pr_pass and ar_pass and physical_pass:
                candidates.append(("one_column", column, factor))
    unique = {(kind, column, factor) for kind, column, factor in candidates}
    return sorted(unique, key=lambda item: (item[0], item[1] or "", item[2]))


def fix_unit_errors(df):
    """Correct uniquely evidenced powers-of-ten errors in mean and worst geometry."""
    result = df.copy()
    result["flag_unit_fixed"] = False
    changed_rows = set()
    case_indices = {"radius": set(), "perimeter": set(), "area": set(), "all_three_scaled": set(), "unrepairable": set()}
    factor_counts = {}
    category_indices = {}
    stopped = []

    _, _, _, _, se_ratio = _ratios(result, "se")
    valid_se = se_ratio.notna()
    se_median = float(se_ratio.loc[valid_se].median()) if valid_se.any() else np.nan
    if valid_se.any() and not bool(_normal_ratio(se_ratio.loc[valid_se], GEOMETRY["area_radius_squared"][0]).mean() >= 0.95):
        stopped.append(
            f"se branch stopped: area_se/radius_se^2 median={se_median:.6g}; "
            f"only {int(_normal_ratio(se_ratio.loc[valid_se], GEOMETRY['area_radius_squared'][0]).sum())}/{int(valid_se.sum())} values match the expected normal band"
        )
    else:
        stopped.append("se branch checked against the expected area/radius-squared relation")

    for suffix in ("mean", "worst"):
        radius_values, _, _, pr_values, ar_values = _ratios(result, suffix)
        pr_normal = _normal_ratio(pr_values, GEOMETRY["perimeter_radius"][0])
        ar_normal = _normal_ratio(ar_values, GEOMETRY["area_radius_squared"][0])
        suspect = (pr_values.notna() & ~pr_normal) | (ar_values.notna() & ~ar_normal)
        suspect |= pr_normal & ar_normal & ~radius_values.between(3, 40)
        for index in result.index[suspect]:
            row = result.loc[[index]]
            candidates = _unit_candidates(row, suffix)
            if len(candidates) == 1:
                kind, column, factor = candidates[0]
                if kind == "all_three_scaled":
                    for measure, divisor in (
                        ("radius", factor),
                        ("perimeter", factor),
                        ("area", factor ** 2),
                    ):
                        name = f"{measure}_{suffix}"
                        result.at[index, name] = result.at[index, name] / divisor
                    case_indices["all_three_scaled"].add(index)
                    category = "all_three_scaled"
                else:
                    result.at[index, column] = result.at[index, column] / factor
                    measure = column.rsplit("_", 1)[0]
                    case_indices[measure].add(index)
                    category = measure
                result.at[index, "flag_unit_fixed"] = True
                changed_rows.add(index)
                factor_counts[str(factor)] = factor_counts.get(str(factor), 0) + 1
                category_indices.setdefault(category, set()).add(index)
            elif len(candidates) > 1:
                case_indices["unrepairable"].add(index)
                category_indices.setdefault("unrepairable", set()).add(index)
            else:
                radius, perimeter, area, pr, ar = _ratios(row, suffix)
                ratio_checks = []
                if pr.notna().iloc[0]:
                    ratio_checks.append(bool(_normal_ratio(pr, GEOMETRY["perimeter_radius"][0]).iloc[0]))
                if ar.notna().iloc[0]:
                    ratio_checks.append(bool(_normal_ratio(ar, GEOMETRY["area_radius_squared"][0]).iloc[0]))
                if ratio_checks and not all(ratio_checks):
                    case_indices["unrepairable"].add(index)
                    category_indices.setdefault("unrepairable", set()).add(index)
                    if len(ratio_checks) == 2:
                        d_pr = int(round(np.log10(pr.iloc[0] / GEOMETRY["perimeter_radius"][0]))) if pd.notna(pr.iloc[0]) else 0
                        d_ar = int(round(np.log10(ar.iloc[0] / GEOMETRY["area_radius_squared"][0]))) if pd.notna(ar.iloc[0]) else 0
                        target = None
                        if d_pr == 0 and d_ar != 0:
                            target = f"area_{suffix}"
                        elif d_pr != 0 and d_ar == 0 and 3 <= radius.iloc[0] <= 40:
                            target = f"perimeter_{suffix}"
                        elif d_pr != 0 and d_ar == 2 * d_pr:
                            target = f"radius_{suffix}"
                        if target:
                            result.at[index, target] = np.nan
                            result.at[index, "flag_unit_fixed"] = True
                            changed_rows.add(index)
                    elif len(ratio_checks) == 1:
                        if pd.notna(ar.iloc[0]) and not _normal_ratio(ar, GEOMETRY["area_radius_squared"][0]).iloc[0]:
                            target = f"radius_{suffix}" if not 3 <= radius.iloc[0] <= 40 else f"area_{suffix}"
                            result.at[index, target] = np.nan
                            result.at[index, "flag_unit_fixed"] = True
                            changed_rows.add(index)
                        elif pd.notna(pr.iloc[0]) and not _normal_ratio(pr, GEOMETRY["perimeter_radius"][0]).iloc[0]:
                            target = f"radius_{suffix}" if not 3 <= radius.iloc[0] <= 40 else f"perimeter_{suffix}"
                            result.at[index, target] = np.nan
                            result.at[index, "flag_unit_fixed"] = True
                            changed_rows.add(index)

    stratified = []
    for category, indices in category_indices.items():
        for field, values in (
            ("center", result.get("center", pd.Series("unknown", index=result.index))),
            ("month", pd.to_datetime(result.get("exam_date", pd.Series(pd.NaT, index=result.index)), errors="coerce").dt.to_period("M").astype("string").fillna("NaT")),
        ):
            counts = values.loc[list(indices)].astype("string").fillna("unknown").value_counts().sort_index()
            stratified.extend(f"{category}/{field}={group}:{int(count)}" for group, count in counts.items())
    detail = "; ".join(f"{name}={len(indices)}" for name, indices in case_indices.items())
    factor_detail = ", ".join(f"{factor}x:{count}" for factor, count in sorted(factor_counts.items())) or "none"
    stratified_detail = "; ".join(stratified) or "none"
    return result, _entry(
        "Power-of-ten geometry unit errors",
        "radius/perimeter/area mean and worst; SE branch assessed separately",
        len(changed_rows),
        f"Rows by case: {detail}; factors: {factor_detail}; centre/month counts: {stratified_detail}; {'. '.join(stopped)}.",
        "Only unique powers-of-ten corrections that restore every available expected geometry ratio were applied; unsupported or ambiguous cases were not guessed.",
    )


def resolve_duplicates(df, has_label):
    """Merge repeated patient IDs and drop only training IDs with conflicting labels."""
    result = df.copy()
    result["flag_merged_duplicate"] = False
    data_columns = [column for column in result.columns if not column.startswith("flag_")]
    exact_before = int(result[data_columns].duplicated().sum())
    repeated_before = int(result["patient_id"].duplicated().sum()) if "patient_id" in result else 0
    repeated = result[result["patient_id"].duplicated(keep=False)] if "patient_id" in result else result.iloc[0:0]
    differences = {}
    label_conflict_patients = set()
    if not repeated.empty:
        for patient_id, copies in repeated.groupby("patient_id", dropna=False, sort=False):
            values = copies[data_columns]
            for column in data_columns:
                if values[column].nunique(dropna=False) > 1:
                    differences[column] = differences.get(column, 0) + 1
            if has_label:
                label_column = "diagnosis" if "diagnosis" in copies else "y"
                if copies[label_column].nunique(dropna=False) > 1:
                    label_conflict_patients.add(patient_id)

    rows_to_keep = []
    rows_removed = 0
    fill_count = 0
    for patient_id, copies in result.groupby("patient_id", dropna=False, sort=False):
        if len(copies) == 1:
            rows_to_keep.append(copies.iloc[0].copy())
            continue
        if has_label and patient_id in label_conflict_patients:
            rows_removed += len(copies)
            continue
        ordered = copies.assign(
            _missing=copies[data_columns].isna().sum(axis=1),
            _date=pd.to_datetime(copies.get("exam_date", pd.Series(pd.NaT, index=copies.index)), errors="coerce"),
            _position=np.arange(len(copies)),
        ).sort_values(["_missing", "_date", "_position"], ascending=[True, False, True], kind="stable")
        merged = ordered.iloc[0][result.columns].copy()
        merged["flag_merged_duplicate"] = True
        for _, other in ordered.iloc[1:].iterrows():
            fillable = merged[data_columns].isna() & other[data_columns].notna()
            fill_count += int(fillable.sum())
            for column in data_columns:
                if pd.isna(merged[column]) and pd.notna(other[column]):
                    merged[column] = other[column]
        rows_removed += len(copies) - 1
        rows_to_keep.append(merged)

    merged_result = pd.DataFrame(rows_to_keep, columns=result.columns).reset_index(drop=True)
    after_count = len(merged_result)
    exact_after = int(merged_result[[c for c in merged_result.columns if not c.startswith("flag_")]].duplicated().sum())
    repeated_after = int(merged_result["patient_id"].duplicated().sum()) if "patient_id" in merged_result else 0
    if not has_label and merged_result["patient_id"].nunique(dropna=False) == 2884:
        assert len(merged_result) == 2884
    changed_rows = rows_removed + int(merged_result["flag_merged_duplicate"].sum())
    difference_detail = ", ".join(f"{column}:{count} patient groups" for column, count in sorted(differences.items())) or "none"
    label_detail = len(label_conflict_patients) if has_label else 0
    conflict_rows = sum(int((result["patient_id"] == pid).sum()) for pid in label_conflict_patients)
    return merged_result, _entry(
        "Repeated patient records",
        "patient_id plus differing columns",
        changed_rows,
        f"Before merge: exact duplicate rows={exact_before}, repeated-ID rows={repeated_before}; differing columns by patient group: {difference_detail}; label-conflict patients={label_detail}; dropped conflicting-label rows={conflict_rows}; merged rows removed={rows_removed}; NaN cells filled={fill_count}; after merge: exact duplicate rows={exact_after}, repeated-ID rows={repeated_after}, rows={after_count}.",
        "One record per patient was retained by fewest NaNs, latest valid exam date, then original order; remaining gaps were filled from copies.",
    )


def check_basic_columns(df):
    """Apply basic age and date bounds and report malformed identifiers."""
    result = df.copy()
    changed_rows = set()
    if "age" in result:
        age = pd.to_numeric(result["age"], errors="coerce")
        bad_age = age.notna() & ~age.between(18, 100)
        result.loc[bad_age, "age"] = np.nan
        changed_rows.update(result.index[bad_age])
    else:
        bad_age = pd.Series(False, index=result.index)
    if "patient_id" in result:
        invalid_id = ~result["patient_id"].astype("string").str.fullmatch(r"\d+", na=False)
        invalid_ids = int(invalid_id.sum())
    else:
        invalid_ids = 0
    if "exam_date" in result:
        dates = pd.to_datetime(result["exam_date"], errors="coerce")
        outside = dates.notna() & ((dates < pd.Timestamp("2024-01-01")) | (dates > pd.Timestamp("2026-12-31")))
        result.loc[outside, "exam_date"] = pd.NaT
        changed_rows.update(result.index[outside])
    else:
        outside = pd.Series(False, index=result.index)
    return result, _entry(
        "Basic age, identifier, and date checks",
        "age, patient_id, exam_date",
        len(changed_rows),
        f"Age outside 18-100 set to NaN={int(bad_age.sum())}; non-digit patient_id reported={invalid_ids}; dates outside 2024-01-01..2026-12-31 set to NaT={int(outside.sum())}.",
        "Only explicit age/date validity bounds were applied; malformed identifiers were reported without guessing a replacement.",
    )


def repair_stage4(df, raw_df, has_label, train_reference=None):
    """Run Stage 4 repairs in order and print a compact final geometry self-check."""
    result = df.copy()
    entries = []
    result, entry = fix_negative_values(result)
    entries.append(entry)
    result, entry = resolve_ambiguous_commas(result, raw_df, train_reference=train_reference)
    entries.append(entry)
    result, entry = fix_unit_errors(result)
    entries.append(entry)
    result, entry = resolve_duplicates(result, has_label)
    entries.append(entry)
    result, entry = check_basic_columns(result)
    entries.append(entry)

    negatives = sum(int(pd.to_numeric(result[column], errors="coerce").lt(0).sum()) for column in MEASUREMENT_COLUMNS if column in result)
    print(f"Stage 4 self-check: remaining negative measurements={negatives}")
    for suffix in SUFFIXES:
        _, _, _, pr, ar = _ratios(result, suffix)
        valid = pr.notna() & ar.notna()
        normal = valid & _normal_ratio(pr, GEOMETRY["perimeter_radius"][0]) & _normal_ratio(ar, GEOMETRY["area_radius_squared"][0])
        share = float(normal.sum() / valid.sum()) if valid.any() else np.nan
        print(f"{suffix} normal geometry ratio share={share:.4f} ({int(normal.sum())}/{int(valid.sum())})")
    for column in ("radius_mean", "perimeter_mean", "area_mean"):
        values = pd.to_numeric(result[column], errors="coerce")
        print(f"{column} min/median/max={values.min()}/{values.median()}/{values.max()}")
    return result, entries