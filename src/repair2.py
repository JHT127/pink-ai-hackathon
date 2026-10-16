"""Stage 8 diagnostics and conservative power-of-ten repairs."""

import numpy as np
import pandas as pd

from src.model import get_features, make_models, metrics, oof_predictions
from src.repair import MEASUREMENTS, MEASUREMENT_COLUMNS


POWER_FACTORS = (0.001, 0.01, 0.1, 10.0, 100.0, 1000.0)
POWER_EXPONENTS = (-3, -2, -1, 1, 2, 3)


def _numeric(values):
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _ratio(numerator, denominator):
    left = _numeric(numerator)
    right = _numeric(denominator)
    valid = left.gt(0) & right.gt(0)
    return left.div(right.where(valid))


def _ratio_profile(ratio):
    log_ratio = np.log10(ratio.where(ratio.gt(0))).replace([np.inf, -np.inf], np.nan)
    observed = log_ratio.dropna()
    if observed.empty:
        return log_ratio, np.nan, (np.nan, np.nan)
    q1, q3 = observed.quantile([0.25, 0.75])
    median = float(observed.median())
    spread = float(q3 - q1)
    return log_ratio, median, (float(q1 - 3 * spread), float(q3 + 3 * spread))


def _group_stats(df, group_values, group_name):
    rows = []
    groups = group_values.astype("string").fillna("unknown")
    overall_medians = {
        column: float(_numeric(df[column]).median())
        for column in MEASUREMENT_COLUMNS
        if _numeric(df[column]).notna().any()
    }
    for group, indices in groups.groupby(groups, sort=True).groups.items():
        for column in MEASUREMENT_COLUMNS:
            values = _numeric(df.loc[indices, column]).dropna()
            q1 = float(values.quantile(0.25)) if len(values) else np.nan
            q3 = float(values.quantile(0.75)) if len(values) else np.nan
            rows.append(
                {
                    "by": group_name,
                    "group": str(group),
                    "column": column,
                    "n": int(len(values)),
                    "median": float(values.median()) if len(values) else np.nan,
                    "overall_median": overall_medians.get(column, np.nan),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1 if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def diagnose_stage8(df):
    """Return worst/mean, mean/SE, and center/quarter diagnostic DataFrames."""
    ordering_rows = []
    center_order_rows = []
    worst_distribution_rows = []
    worst_jump_rows = []
    worst_summary_rows = []
    se_distribution_rows = []
    se_jump_rows = []
    se_summary_rows = []
    centers = df.get("center", pd.Series("unknown", index=df.index)).astype("string").fillna("unknown")

    for measurement in MEASUREMENTS:
        mean = _numeric(df[f"{measurement}_mean"])
        worst = _numeric(df[f"{measurement}_worst"])
        valid_pair = mean.notna() & worst.notna()
        below = valid_pair & worst.lt(mean)
        ordering_rows.append(
            {
                "measurement": measurement,
                "n_valid": int(valid_pair.sum()),
                "worst_below_mean": int(below.sum()),
                "share_below_mean": float(below.sum() / valid_pair.sum()) if valid_pair.any() else np.nan,
            }
        )
        for center in sorted(centers.unique()):
            group_mask = centers.eq(center)
            denominator = int((valid_pair & group_mask).sum())
            count = int((below & group_mask).sum())
            center_order_rows.append(
                {
                    "measurement": measurement,
                    "center": center,
                    "n_valid": denominator,
                    "worst_below_mean": count,
                    "share_below_mean": count / denominator if denominator else np.nan,
                }
            )

        worst_ratio = _ratio(worst, mean)
        worst_log = np.log10(worst_ratio.where(worst_ratio.gt(0))).replace([np.inf, -np.inf], np.nan)
        for value, count in worst_log.round(1).value_counts(dropna=True).sort_index().items():
            worst_distribution_rows.append(
                {"measurement": measurement, "log10_ratio_rounded": float(value), "rows": int(count)}
            )
        worst_jumps = _decade_jump_counts(worst_ratio)
        worst_jump_rows.extend({"measurement": measurement, **row} for row in worst_jumps)
        ratio_valid = worst_ratio.notna()
        high = ratio_valid & worst_ratio.gt(5)
        low = ratio_valid & worst_ratio.lt(0.9)
        worst_summary_rows.append(
            {
                "measurement": measurement,
                "n_valid": int(ratio_valid.sum()),
                "ratio_gt_5": int(high.sum()),
                "share_gt_5": float(high.sum() / ratio_valid.sum()) if ratio_valid.any() else np.nan,
                "ratio_lt_0_9": int(low.sum()),
                "share_lt_0_9": float(low.sum() / ratio_valid.sum()) if ratio_valid.any() else np.nan,
                "clean_decade_jumps": int(sum(row["rows"] for row in worst_jumps)),
            }
        )

        se_ratio = _ratio(mean, df[f"{measurement}_se"])
        se_log = np.log10(se_ratio.where(se_ratio.gt(0))).replace([np.inf, -np.inf], np.nan)
        for value, count in se_log.round(1).value_counts(dropna=True).sort_index().items():
            se_distribution_rows.append(
                {"measurement": measurement, "log10_ratio_rounded": float(value), "rows": int(count)}
            )
        se_jumps = _decade_jump_counts(se_ratio)
        se_jump_rows.extend({"measurement": measurement, **row} for row in se_jumps)
        valid_se = se_ratio.dropna()
        if len(valid_se):
            p01, p99 = valid_se.quantile([0.01, 0.99])
            outside = valid_se.lt(p01) | valid_se.gt(p99)
            outside_count = int(outside.sum())
        else:
            p01, p99, outside_count = np.nan, np.nan, 0
        se_summary_rows.append(
            {
                "measurement": measurement,
                "n_valid": int(len(valid_se)),
                "ratio_p01": float(p01),
                "ratio_p99": float(p99),
                "outside_p01_p99": outside_count,
                "share_outside_p01_p99": outside_count / len(valid_se) if len(valid_se) else np.nan,
                "clean_decade_jumps": int(sum(row["rows"] for row in se_jumps)),
            }
        )

    dates = pd.to_datetime(df.get("exam_date", pd.Series(pd.NaT, index=df.index)), errors="coerce")
    quarter = dates.dt.to_period("Q").astype("string").fillna("unknown")
    center_stats = _group_stats(df, centers, "center")
    quarter_stats = _group_stats(df, quarter, "quarter")
    center_findings = _center_scale_findings(center_stats)
    quarter_findings = _center_scale_findings(quarter_stats)
    center_offset_findings = _center_offset_findings(center_stats)
    quarter_offset_findings = _center_offset_findings(quarter_stats)
    return {
        "worst_mean": pd.DataFrame(ordering_rows),
        "worst_mean_by_center": pd.DataFrame(center_order_rows),
        "worst_mean_ratio_distribution": pd.DataFrame(worst_distribution_rows),
        "worst_mean_decade_jumps": pd.DataFrame(worst_jump_rows),
        "worst_mean_ratio_summary": pd.DataFrame(worst_summary_rows),
        "mean_se_ratio_distribution": pd.DataFrame(se_distribution_rows),
        "mean_se_decade_jumps": pd.DataFrame(se_jump_rows),
        "mean_se_ratio_summary": pd.DataFrame(se_summary_rows),
        "by_center": center_stats,
        "by_quarter": quarter_stats,
        "center_scale_findings": center_findings,
        "quarter_scale_findings": quarter_findings,
        "center_offset_findings": center_offset_findings,
        "quarter_offset_findings": quarter_offset_findings,
    }


def _decade_jump_counts(ratio):
    log_ratio, median, bounds = _ratio_profile(ratio)
    delta = log_ratio - median
    rounded = delta.round().astype("Int64")
    outside = ~log_ratio.between(*bounds)
    clean = outside & delta.sub(rounded.astype(float)).abs().le(0.1) & rounded.isin(POWER_EXPONENTS)
    counts = rounded.loc[clean].value_counts().sort_index()
    return [
        {"power_exponent": int(exponent), "rows": int(count)}
        for exponent, count in counts.items()
    ]


def _center_scale_findings(stats):
    """Flag group medians separated from the overall median by clean decades."""
    rows = []
    if stats.empty:
        return pd.DataFrame(rows)
    for column, column_stats in stats.groupby("column", sort=False):
        overall = float(column_stats["overall_median"].iloc[0])
        if not np.isfinite(overall) or overall <= 0:
            continue
        for row in column_stats.itertuples(index=False):
            median = row.median
            if row.n < 100 or not np.isfinite(median) or median <= 0:
                continue
            log_difference = np.log10(median / overall)
            exponent = int(round(log_difference))
            if exponent in POWER_EXPONENTS and abs(log_difference - exponent) <= 0.1:
                rows.append(
                    {
                        "by": row.by,
                        "group": row.group,
                        "column": column,
                        "group_median": median,
                        "overall_median": overall,
                        "power_exponent": exponent,
                        "factor_to_overall": 10.0 ** (-exponent),
                    }
                )
    return pd.DataFrame(rows)


def _center_offset_findings(stats):
    """Flag group median offsets matching a signed clean power of ten."""
    rows = []
    if stats.empty:
        return pd.DataFrame(rows)
    for row in stats.itertuples(index=False):
        overall = row.overall_median
        median = row.median
        if row.n < 100 or not np.isfinite(overall) or not np.isfinite(median):
            continue
        difference = median - overall
        for factor in POWER_FACTORS:
            if abs(abs(difference) - factor) <= factor * 0.01:
                rows.append(
                    {
                        "by": row.by,
                        "group": row.group,
                        "column": row.column,
                        "group_median": median,
                        "overall_median": overall,
                        "offset": float(np.sign(difference) * factor),
                    }
                )
                break
    return pd.DataFrame(rows)


def confident_oof_mistakes(df, limit=40):
    """Return the most confident base-HGB OOF mistakes and their center/flag shares."""
    features = get_features(df)
    target = df["y"].astype(int)
    groups = df["patient_id"]
    probability = oof_predictions({"hgb": make_models("base")["hgb"]}, features, target, groups)["hgb"]
    labels = target.to_numpy()
    mistakes = ((labels == 0) & (probability > 0.9)) | ((labels == 1) & (probability < 0.1))
    positions = np.flatnonzero(mistakes)
    confidence = np.where(labels[positions] == 0, probability[positions], 1 - probability[positions])
    selected_order = np.argsort(-confidence, kind="stable")[:limit]
    selected_positions = positions[selected_order]
    selected_index = df.index[selected_positions]

    missing = features.isna().sum(axis=1)
    result = df.loc[selected_index, ["patient_id", "center", "y"]].copy()
    result["probability"] = probability[selected_positions]
    result["n_missing"] = missing.loc[selected_index].to_numpy()
    result["flag_unit_fixed"] = df.get("flag_unit_fixed", pd.Series(False, index=df.index)).loc[selected_index].fillna(False).to_numpy()
    for measurement in MEASUREMENTS:
        result[f"{measurement}_mean"] = _numeric(df.loc[selected_index, f"{measurement}_mean"]).to_numpy()
    result["confidence"] = confidence[selected_order]

    all_mistakes = pd.DataFrame(
        {
            "center": df.get("center", pd.Series("unknown", index=df.index)).astype("string").fillna("unknown").to_numpy()[positions],
            "flag_unit_fixed": df.get("flag_unit_fixed", pd.Series(False, index=df.index)).fillna(False).to_numpy()[positions],
        }
    )
    center_share = all_mistakes.groupby("center", dropna=False).size().rename("mistakes").reset_index()
    center_share["share_of_mistakes"] = center_share["mistakes"] / max(len(positions), 1)
    flag_share = all_mistakes.groupby("flag_unit_fixed", dropna=False).size().rename("mistakes").reset_index()
    flag_share["share_of_mistakes"] = flag_share["mistakes"] / max(len(positions), 1)
    return result.sort_values("confidence", ascending=False).reset_index(drop=True), center_share, flag_share


def _reference_profiles(train_reference, pair):
    first_column, second_column = pair
    first = _numeric(train_reference[first_column])
    second = _numeric(train_reference[second_column])
    ratio = _ratio(first, second)
    log_ratio, median, bounds = _ratio_profile(ratio)
    normal = log_ratio.between(*bounds) if np.isfinite(bounds[0]) else pd.Series(False, index=ratio.index)
    ranges = {}
    medians = {}
    spreads = {}
    for column, values in ((first_column, first), (second_column, second)):
        reference = values.loc[normal & values.gt(0)].dropna()
        if len(reference):
            ranges[column] = (float(reference.quantile(0.01)), float(reference.quantile(0.99)))
            medians[column] = float(reference.median())
            q1, q3 = reference.quantile([0.25, 0.75])
            spreads[column] = float(q3 - q1)
        else:
            ranges[column] = (np.nan, np.nan)
            medians[column] = np.nan
            spreads[column] = np.nan
    return median, bounds, ranges, medians, spreads


def _inside(value, bounds):
    return np.isfinite(value) and np.isfinite(bounds[0]) and bounds[0] <= value <= bounds[1]


def _distance_outside(value, bounds, spread):
    if not np.isfinite(value) or not np.isfinite(bounds[0]) or bounds[0] <= value <= bounds[1]:
        return 0.0
    scale = spread if np.isfinite(spread) and spread > 0 else max(abs(bounds[0]), abs(bounds[1]), 1e-12)
    return (bounds[0] - value if value < bounds[0] else value - bounds[1]) / scale


def _repair_pair(result, train_reference, measurement, pair, counts, changed_columns):
    first_column, second_column = pair
    median, bounds, ranges, medians, spreads = _reference_profiles(train_reference, pair)
    if not np.isfinite(median):
        return
    current_ratio = _ratio(result[first_column], result[second_column])
    current_log = np.log10(current_ratio.where(current_ratio.gt(0))).replace([np.inf, -np.inf], np.nan)
    for index in result.index[current_log.notna()]:
        observed_log = float(current_log.loc[index])
        if bounds[0] <= observed_log <= bounds[1]:
            continue
        exponent = int(round(observed_log - median))
        if exponent not in POWER_EXPONENTS or abs((observed_log - median) - exponent) > 0.1:
            continue
        original = {
            first_column: float(_numeric(pd.Series([result.at[index, first_column]])).iloc[0]),
            second_column: float(_numeric(pd.Series([result.at[index, second_column]])).iloc[0]),
        }
        candidates = []
        for column in (first_column, second_column):
            for factor in POWER_FACTORS:
                corrected_value = original[column] * factor
                corrected_ratio = original[first_column] / original[second_column]
                corrected_ratio *= factor if column == first_column else 1 / factor
                corrected_log = np.log10(corrected_ratio) if corrected_ratio > 0 else np.nan
                if _inside(corrected_value, ranges[column]) and bounds[0] <= corrected_log <= bounds[1]:
                    spread = spreads[column]
                    scale = spread if np.isfinite(spread) and spread > 0 else max(abs(medians[column]), 1e-12)
                    typical_distance = abs(corrected_value - medians[column]) / scale
                    candidates.append((column, factor, typical_distance))

        columns = {column for column, _, _ in candidates}
        chosen_column = None
        if len(candidates) == 1:
            column, factor, _ = candidates[0]
            result.at[index, column] = original[column] * factor
            counts[measurement]["repaired"] += 1
            changed_columns.add((index, column))
            continue
        if candidates and len(columns) == 1:
            chosen_column = next(iter(columns))
        elif candidates and len(columns) > 1:
            best_by_column = sorted(
                (min(distance for candidate_column, _, distance in candidates if candidate_column == column), column)
                for column in columns
            )
            if best_by_column[1][0] - best_by_column[0][0] >= 0.1:
                chosen_column = best_by_column[0][1]
            else:
                outside_reference = [
                    column
                    for column in columns
                    if not _inside(original[column], ranges[column])
                ]
                if len(outside_reference) == 1:
                    chosen_column = outside_reference[0]
        elif not candidates:
            ranked = sorted(
                (_distance_outside(original[column], ranges[column], spreads[column]), column)
                for column in (first_column, second_column)
            )
            if ranked[-1][0] > ranked[0][0] and ranked[-1][0] > 0:
                chosen_column = ranked[-1][1]

        if chosen_column is None:
            counts[measurement]["unresolved"] += 1
        else:
            result.at[index, chosen_column] = np.nan
            counts[measurement]["set_nan"] += 1
            changed_columns.add((index, chosen_column))


def _center_scale_rules(train_reference):
    if "center" not in train_reference:
        return []
    centers = train_reference["center"].astype("string").fillna("unknown")
    rules = []
    for column in MEASUREMENT_COLUMNS:
        values = _numeric(train_reference[column])
        overall_median = float(values.median()) if values.notna().any() else np.nan
        if not np.isfinite(overall_median) or overall_median <= 0:
            continue
        low, high = values.quantile([0.01, 0.99])
        for center in sorted(centers.unique()):
            center_values = values.loc[centers.eq(center)].dropna()
            if len(center_values) < 100:
                continue
            center_median = float(center_values.median())
            ratio = center_median / overall_median
            exponent = int(round(np.log10(ratio))) if ratio > 0 else 0
            if exponent not in POWER_EXPONENTS or abs(np.log10(ratio) - exponent) > 0.1:
                continue
            factor = 10.0 ** (-exponent)
            if low <= center_median * factor <= high:
                rules.append((center, column, factor, center_median, overall_median, float(low), float(high)))
    return rules


def _center_offset_rules(train_reference):
    if "center" not in train_reference:
        return []
    centers = train_reference["center"].astype("string").fillna("unknown")
    rules = []
    for column in MEASUREMENT_COLUMNS:
        values = _numeric(train_reference[column])
        overall_median = float(values.median()) if values.notna().any() else np.nan
        if not np.isfinite(overall_median):
            continue
        low, high = values.quantile([0.01, 0.99])
        for center in sorted(centers.unique()):
            center_values = values.loc[centers.eq(center)].dropna()
            if len(center_values) < 100:
                continue
            center_median = float(center_values.median())
            difference = center_median - overall_median
            for factor in POWER_FACTORS:
                if abs(abs(difference) - factor) <= factor * 0.01:
                    offset = float(np.sign(difference) * factor)
                    if low <= center_median - offset <= high:
                        rules.append((center, column, offset, center_median, overall_median, float(low), float(high)))
                    break
    return rules


def repair_stage8(df, train_reference=None):
    """Apply clean-decade repairs using train-derived ratio and value ranges."""
    reference = df if train_reference is None else train_reference
    result = df.copy()
    result["flag_stage8"] = False
    counts = {name: {"repaired": 0, "set_nan": 0, "unresolved": 0} for name in MEASUREMENTS}
    changed_columns = set()

    for measurement in MEASUREMENTS:
        _repair_pair(
            result,
            reference,
            measurement,
            (f"{measurement}_worst", f"{measurement}_mean"),
            counts,
            changed_columns,
        )
        if _decade_jump_counts(_ratio(reference[f"{measurement}_mean"], reference[f"{measurement}_se"])):
            _repair_pair(
                result,
                reference,
                measurement,
                (f"{measurement}_mean", f"{measurement}_se"),
                counts,
                changed_columns,
            )

    center_rule_rows = []
    if "center" in result:
        centers = result["center"].astype("string").fillna("unknown")
        for center, column, factor, center_median, overall_median, low, high in _center_scale_rules(reference):
            repaired = 0
            set_nan = 0
            for index in result.index[centers.eq(center)]:
                value = _numeric(pd.Series([result.at[index, column]])).iloc[0]
                if pd.isna(value):
                    continue
                candidate = float(value) * factor
                if low <= candidate <= high:
                    result.at[index, column] = candidate
                    changed_columns.add((index, column))
                    repaired += 1
                elif not low <= float(value) <= high:
                    result.at[index, column] = np.nan
                    changed_columns.add((index, column))
                    set_nan += 1
            center_rule_rows.append(
                {
                    "center": center,
                    "column": column,
                    "factor": factor,
                    "center_median": center_median,
                    "overall_median": overall_median,
                    "repaired": repaired,
                    "set_nan": set_nan,
                }
            )
            measurement = column.rsplit("_", 1)[0]
            counts[measurement]["repaired"] += repaired
            counts[measurement]["set_nan"] += set_nan

        for center, column, offset, center_median, overall_median, low, high in _center_offset_rules(reference):
            repaired = 0
            set_nan = 0
            for index in result.index[centers.eq(center)]:
                value = _numeric(pd.Series([result.at[index, column]])).iloc[0]
                if pd.isna(value):
                    continue
                candidate = float(value) - offset
                if low <= candidate <= high:
                    result.at[index, column] = candidate
                    changed_columns.add((index, column))
                    repaired += 1
                elif not low <= float(value) <= high:
                    result.at[index, column] = np.nan
                    changed_columns.add((index, column))
                    set_nan += 1
            center_rule_rows.append(
                {
                    "center": center,
                    "column": column,
                    "offset": offset,
                    "center_median": center_median,
                    "overall_median": overall_median,
                    "repaired": repaired,
                    "set_nan": set_nan,
                }
            )
            measurement = column.rsplit("_", 1)[0]
            counts[measurement]["repaired"] += repaired
            counts[measurement]["set_nan"] += set_nan

    for index, _ in changed_columns:
        result.at[index, "flag_stage8"] = True

    for measurement in MEASUREMENTS:
        counts[measurement]["repaired"] = 0
        counts[measurement]["set_nan"] = 0
    for index, column in changed_columns:
        measurement = column.rsplit("_", 1)[0]
        outcome = "set_nan" if pd.isna(result.at[index, column]) else "repaired"
        counts[measurement][outcome] += 1

    per_measurement = pd.DataFrame(
        [{"measurement": name, **value} for name, value in counts.items()]
    )
    center_rows = []
    center_values = result.get("center", pd.Series("unknown", index=result.index)).astype("string").fillna("unknown")
    for measurement in MEASUREMENTS:
        columns = [f"{measurement}_{suffix}" for suffix in ("mean", "se", "worst")]
        for center in sorted(center_values.unique()):
            indices = result.index[center_values.eq(center)]
            repaired = 0
            set_nan = 0
            for index in indices:
                for column in columns:
                    if (index, column) not in changed_columns:
                        continue
                    if pd.isna(result.at[index, column]):
                        set_nan += 1
                    else:
                        repaired += 1
            center_rows.append(
                {"measurement": measurement, "center": center, "repaired": repaired, "set_nan": set_nan}
            )

    center_detail = "; ".join(
        f"{row['center']}/{row['column']} adjustment={row.get('factor', row.get('offset'))}: repaired={row['repaired']}, NaN={row['set_nan']}"
        for row in center_rule_rows
    ) or "none"
    detail = "; ".join(
        f"{row['measurement']}: repaired={row['repaired']}, NaN={row['set_nan']}, unresolved={row['unresolved']}"
        for row in per_measurement.to_dict("records")
    )
    rows_affected = len({index for index, _ in changed_columns})
    entries = [
        {
            "problem": "Clean power-of-ten measurement inconsistencies",
            "columns": ", ".join(MEASUREMENT_COLUMNS),
            "rows_affected": rows_affected,
            "fix": f"Per measurement: {detail}. Center-scale findings: {center_detail}.",
            "why": "Changes require a clean decade discrepancy, a normal corrected ratio, and a train-derived 1st-99th percentile value range; unsupported rows are not guessed.",
        },
        {
            "problem": "Stage 8 repair counts by measurement",
            "columns": ", ".join(MEASUREMENT_COLUMNS),
            "rows_affected": rows_affected,
            "fix": per_measurement.to_dict("records"),
            "why": "Repair and NaN counts are reported separately; unresolved anomalies remain unchanged.",
        },
        {
            "problem": "Stage 8 repair counts by center",
            "columns": ", ".join(MEASUREMENT_COLUMNS),
            "rows_affected": rows_affected,
            "fix": center_rows,
            "why": "Center totals are cell-level counts from the same train-derived rules.",
        },
    ]
    result.attrs["stage8_per_measurement"] = per_measurement
    result.attrs["stage8_by_center"] = pd.DataFrame(center_rows)
    result.attrs["stage8_center_rules"] = pd.DataFrame(center_rule_rows)
    return result, entries


def compare_stage4_stage8(train_stage4, train_stage8):
    """Compare base HGB/logistic OOF metrics on the same patient-grouped folds."""
    rows = []
    predictions = {}
    for stage, frame in (("stage4", train_stage4), ("stage8", train_stage8)):
        features = get_features(frame)
        target = frame["y"].astype(int)
        groups = frame["patient_id"]
        fold_predictions = oof_predictions(
            {name: make_models("base")[name] for name in ("hgb", "logreg")},
            features,
            target,
            groups,
        )
        for model_name, probability in fold_predictions.items():
            predictions[(stage, model_name)] = probability
            score = metrics(target, probability)
            rows.append(
                {
                    "stage": stage,
                    "model": model_name,
                    "roc_auc": score["roc_auc"],
                    "brier": score["brier_score"],
                    "recall_at_precision_0_90": score["recall_at_precision"],
                }
            )
    return pd.DataFrame(rows), predictions


def worst_mean_share(df):
    """Return pair-level and row-level worst/mean ordering shares."""
    valid_count = 0
    ordered_count = 0
    row_valid = pd.Series(False, index=df.index)
    row_ordered = pd.Series(True, index=df.index)
    for measurement in MEASUREMENTS:
        mean = _numeric(df[f"{measurement}_mean"])
        worst = _numeric(df[f"{measurement}_worst"])
        valid = mean.notna() & worst.notna()
        ordered = worst.ge(mean).where(valid, True)
        valid_count += int(valid.sum())
        ordered_count += int((valid & ordered).sum())
        row_valid |= valid
        row_ordered &= ordered
    return {
        "ordered_pairs": ordered_count,
        "valid_pairs": valid_count,
        "pair_share": ordered_count / valid_count if valid_count else np.nan,
        "rows_all_observed_ordered": int((row_valid & row_ordered).sum()),
        "rows_with_any_observed_pair": int(row_valid.sum()),
        "row_share": float((row_valid & row_ordered).sum() / row_valid.sum()) if row_valid.any() else np.nan,
    }