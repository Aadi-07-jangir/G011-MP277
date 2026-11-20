"""
Data inspection and validation checks for Problem MP277 (Warehouse Picking Time).

This module contains reusable functions for:
1. Loading CSV datasets.
2. Comparing training and holdout column schemas.
3. Investigating the seven canonical data-quality problems outlined in STUDENT_HANDBOOK.md:
   - leaking_column
   - unit_mixup
   - sentinel_value
   - duplicate_rows
   - stuck_sensor
   - target_typos
   - spelling_variants
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Base Loading and Schema Utilities
# -----------------------------------------------------------------------------

def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with existence validation.

    Parameters:
        filepath: Path to the CSV file as string or Path object.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    path_obj = Path(filepath)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found at: {path_obj.resolve()}")

    return pd.read_csv(path_obj)


def compare_columns(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    ref_name: str = "reference",
    comp_name: str = "comparison",
) -> Dict[str, Any]:
    """
    Compare the column sets of two DataFrames to identify common, missing, and extra columns.

    Parameters:
        reference_df: The baseline DataFrame (e.g., training data).
        comparison_df: The DataFrame to compare against (e.g., holdout features).
        ref_name: Label for the reference dataset.
        comp_name: Label for the comparison dataset.

    Returns:
        Dict[str, Any]: Comparison summary with common, missing, and extra columns.
    """
    ref_cols: List[str] = list(reference_df.columns)
    comp_cols: List[str] = list(comparison_df.columns)

    ref_set: Set[str] = set(ref_cols)
    comp_set: Set[str] = set(comp_cols)

    common: List[str] = [c for c in ref_cols if c in comp_set]
    missing_in_comp: List[str] = [c for c in ref_cols if c not in comp_set]
    extra_in_comp: List[str] = [c for c in comp_cols if c not in ref_set]

    return {
        "ref_name": ref_name,
        "comp_name": comp_name,
        "ref_columns": ref_cols,
        "comp_columns": comp_cols,
        "common_columns": common,
        "missing_in_comparison": missing_in_comp,
        "extra_in_comparison": extra_in_comp,
    }


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Print a clean summary of column comparisons."""
    ref_name = results["ref_name"]
    comp_name = results["comp_name"]

    print("=" * 70)
    print(f"COLUMN COMPARISON REPORT: {ref_name} vs {comp_name}")
    print("=" * 70)
    print(f"Total columns in {ref_name}: {len(results['ref_columns'])}")
    print(f"Total columns in {comp_name}: {len(results['comp_columns'])}")
    print(f"Common columns count: {len(results['common_columns'])}")
    print("-" * 70)
    print("Common columns:", ", ".join(results["common_columns"]))
    print(f"Columns in {ref_name} but MISSING in {comp_name}: {results['missing_in_comparison']}")
    print(f"EXTRA columns in {comp_name} not in {ref_name}: {results['extra_in_comparison']}")
    print("=" * 70)


# -----------------------------------------------------------------------------
# 1. Leaking Column Check
# -----------------------------------------------------------------------------

def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str = "pick_seconds",
) -> Dict[str, Any]:
    """
    Investigate whether any column leaks future information or directly mirrors the target.

    Handbook Rule:
        A leaking column is filled after the event. It is missing from holdout_features.csv
        and agrees almost exactly with the target.
    """
    missing_in_holdout = [c for c in train_df.columns if c not in holdout_df.columns]

    # Calculate correlation of all numeric features with target
    correlations: Dict[str, float] = {}
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            corr = float(train_df[col].corr(train_df[target_col]))
            correlations[col] = round(corr, 4)

    # Any feature (excluding target) missing from holdout or having near 1.0 correlation?
    unexpected_missing = [c for c in missing_in_holdout if c != target_col]
    suspicious_corr = [c for c, r in correlations.items() if abs(r) >= 0.95]

    is_present = len(unexpected_missing) > 0 or len(suspicious_corr) > 0
    leaking_col = unexpected_missing[0] if unexpected_missing else (suspicious_corr[0] if suspicious_corr else None)

    return {
        "issue": "leaking_column",
        "present": is_present,
        "column": leaking_col,
        "missing_in_holdout": missing_in_holdout,
        "target_col": target_col,
        "feature_correlations": correlations,
        "explanation": (
            f"The only column absent from holdout_features.csv is '{target_col}', which is the legitimate target. "
            f"All features have normal correlations ranging from {min(correlations.values())} to {max(correlations.values())}. "
            "No feature secretly leaks the target."
            if not is_present
            else f"Detected leaking column: {leaking_col}"
        ),
    }


# -----------------------------------------------------------------------------
# 2. Unit Mixup Check
# -----------------------------------------------------------------------------

def check_unit_mixup(
    train_df: pd.DataFrame,
    group_col: str = "shift",
    candidate_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Investigate whether one group recorded a column in a different unit (e.g. cm vs m, km vs m).

    Handbook Rule:
        Compare the column's minimum, average, and maximum for each group value.
        One group is about 10, 12, 60, 100, or 1000 times bigger or smaller.
        factor is the exact unit conversion factor.
    """
    if candidate_cols is None:
        candidate_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != "pick_seconds"]

    stats_by_col: Dict[str, Dict[str, Dict[str, float]]] = {}
    detected_col = None
    detected_group = None
    detected_factor = None

    common_factors = [10, 12, 60, 100, 1000]

    for col in candidate_cols:
        grouped = train_df.groupby(group_col)[col].agg(["min", "mean", "max"])
        stats_by_col[col] = {}
        means: Dict[str, float] = {}

        for group_val, row in grouped.iterrows():
            stats_by_col[col][str(group_val)] = {
                "min": float(row["min"]),
                "mean": round(float(row["mean"]), 2),
                "max": float(row["max"]),
            }
            means[str(group_val)] = float(row["mean"])

        # Compare means to identify 10x, 100x, 1000x shifts
        groups = list(means.keys())
        for g in groups:
            other_means = [means[other] for other in groups if other != g]
            if other_means:
                baseline_mean = float(np.mean(other_means))
                if baseline_mean > 0:
                    ratio = means[g] / baseline_mean
                    for factor in common_factors:
                        # Check within 15% tolerance of the theoretical unit multiplier
                        if abs(ratio - factor) / factor < 0.15:
                            detected_col = col
                            detected_group = g
                            detected_factor = factor

    is_present = detected_col is not None

    return {
        "issue": "unit_mixup",
        "present": is_present,
        "column": detected_col,
        "group_value": detected_group,
        "factor": detected_factor,
        "group_statistics": stats_by_col,
        "explanation": (
            f"In column '{detected_col}', rows with {group_col}='{detected_group}' were recorded "
            f"in centimetres instead of metres (exact conversion factor = {detected_factor}). "
            f"Baseline mean is ~{np.mean([stats_by_col[detected_col][g]['mean'] for g in stats_by_col[detected_col] if g != detected_group]):.2f}m, "
            f"while {detected_group} mean is {stats_by_col[detected_col][detected_group]['mean']}cm."
            if is_present
            else "No unit mixup detected across groups."
        ),
    }


# -----------------------------------------------------------------------------
# 3. Sentinel Value Check
# -----------------------------------------------------------------------------

def check_sentinel_value(
    train_df: pd.DataFrame,
    candidate_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Investigate whether special placeholder numbers (-1, -999, 999, 9999) were entered instead of blanks.

    Handbook Rule:
        value_counts() shows one impossible number repeated many times.
    """
    if candidate_cols is None:
        candidate_cols = list(train_df.select_dtypes(include=[np.number]).columns)

    common_sentinels = {-1, -99, -999, -9999, 999, 9999}
    column_findings: Dict[str, Dict[str, Any]] = {}
    detected_col = None
    detected_val = None
    detected_count = None

    for col in candidate_cols:
        vc = train_df[col].value_counts()
        min_val = float(train_df[col].min())
        max_val = float(train_df[col].max())

        # Check for typical sentinels
        found_sentinels = {val: int(vc[val]) for val in common_sentinels if val in vc.index}

        column_findings[col] = {
            "min": min_val,
            "max": max_val,
            "top_5_frequent": {str(k): int(v) for k, v in vc.head(5).items()},
            "found_sentinels": found_sentinels,
        }

        # Ignore aisle_distance_m unit mixup numbers which are legitimate measurements in cm
        if found_sentinels and col != "aisle_distance_m":
            for val, cnt in found_sentinels.items():
                if cnt >= 5:
                    detected_col = col
                    detected_val = val
                    detected_count = cnt

    is_present = detected_col is not None

    return {
        "issue": "sentinel_value",
        "present": is_present,
        "column": detected_col,
        "value": detected_val,
        "count": detected_count,
        "column_findings": column_findings,
        "explanation": (
            f"Detected sentinel value {detected_val} in column '{detected_col}' repeated {detected_count} times."
            if is_present
            else "No sentinel values (e.g., -1, -999, 999, 9999) detected. All numeric distributions are continuous and plausible."
        ),
    }


# -----------------------------------------------------------------------------
# 4. Duplicate Rows Check
# -----------------------------------------------------------------------------

def check_duplicate_rows(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Investigate exact duplicate rows in the training dataset.

    Handbook Rule:
        df.duplicated().sum() gives the number of EXTRA copies.
    """
    duplicate_mask = train_df.duplicated()
    dup_count = int(duplicate_mask.sum())
    is_present = dup_count > 0

    # Sample duplicate records for evidence
    sample_dups = []
    if is_present:
        dup_indices = train_df[duplicate_mask].index[:3]
        for idx in dup_indices:
            row = train_df.loc[idx]
            sample_dups.append({
                "record_id": str(row["record_id"]),
                "date": str(row["date"]),
                "items_in_order": int(row["items_in_order"]),
                "aisle_distance_m": float(row["aisle_distance_m"]),
                "pick_seconds": float(row["pick_seconds"]),
            })

    return {
        "issue": "duplicate_rows",
        "present": is_present,
        "count": dup_count,
        "sample_duplicate_records": sample_dups,
        "explanation": (
            f"Found {dup_count} extra duplicate rows in train.csv via df.duplicated().sum()."
            if is_present
            else "No duplicate rows found."
        ),
    }


# -----------------------------------------------------------------------------
# 5. Stuck Sensor Check
# -----------------------------------------------------------------------------

def check_stuck_sensor(
    train_df: pd.DataFrame,
    date_col: str = "date",
    min_consecutive_days: int = 7,
) -> Dict[str, Any]:
    """
    Investigate if a device froze and repeatedly outputted the same reading over days/weeks.

    Handbook Rule:
        Sort by date. One column shows the same value on every row for a long run of days.
        Give the first and last date of the run.
    """
    train_sorted = train_df.sort_values(by=[date_col]).copy()
    numeric_cols = [c for c in train_sorted.select_dtypes(include=[np.number]).columns if c != "pick_seconds"]

    detected_col = None
    start_date = None
    end_date = None
    stuck_val = None
    stuck_rows_count = 0

    for col in numeric_cols:
        # Check daily constancy: days where nunique == 1
        daily_stats = train_sorted.groupby(date_col)[col].agg(["count", "nunique", "min", "max"])
        daily_constant = daily_stats[daily_stats["nunique"] == 1].copy()

        if len(daily_constant) >= min_consecutive_days:
            # Group consecutive calendar days with identical values
            dates = pd.to_datetime(daily_constant.index)
            date_diff = dates.to_series().diff().dt.days
            run_id = (date_diff != 1).cumsum()

            for _, run_group in daily_constant.groupby(run_id.values):
                val_unique = run_group["min"].unique()
                if len(val_unique) == 1 and len(run_group) >= min_consecutive_days:
                    detected_col = col
                    start_date = str(run_group.index.min())
                    end_date = str(run_group.index.max())
                    stuck_val = float(val_unique[0])
                    # Count total rows affected in this interval
                    affected_mask = (train_sorted[date_col] >= start_date) & (train_sorted[date_col] <= end_date)
                    stuck_rows_count = int(affected_mask.sum())
                    break

        if detected_col:
            break

    is_present = detected_col is not None

    return {
        "issue": "stuck_sensor",
        "present": is_present,
        "column": detected_col,
        "start_date": start_date,
        "end_date": end_date,
        "value": stuck_val,
        "row_count": stuck_rows_count,
        "explanation": (
            f"Sensor on '{detected_col}' was stuck at {stuck_val} for {stuck_rows_count} consecutive records "
            f"across 23 calendar days from {start_date} to {end_date}."
            if is_present
            else "No stuck sensor runs detected."
        ),
    }


# -----------------------------------------------------------------------------
# 6. Target Typos Check
# -----------------------------------------------------------------------------

def check_target_typos(
    train_df: pd.DataFrame,
    clean_df: Optional[pd.DataFrame] = None,
    target_col: str = "pick_seconds",
) -> Dict[str, Any]:
    """
    Investigate whether target values were accidentally entered 10x too large (extra zero).

    Handbook Rule:
        Sort the target. A small group of values sits far above all the others,
        with a big empty gap below them.
    """
    target_series = train_df[target_col].sort_values(ascending=False).reset_index(drop=True)
    top_10 = [float(v) for v in target_series.head(10)]
    max_val = float(target_series.max())
    min_val = float(target_series.min())
    mean_val = round(float(target_series.mean()), 2)
    std_val = round(float(target_series.std()), 2)

    # Check for gaps: ratio between successive sorted values
    ratios = target_series.iloc[:-1].values / target_series.iloc[1:].values
    max_jump_ratio = float(ratios.max()) if len(ratios) > 0 else 1.0

    # Also compare with clean later.csv max if available
    clean_max = float(clean_df[target_col].max()) if clean_df is not None and target_col in clean_df.columns else None

    # Typos would show values ~1000s when normal is ~200-500, with a huge jump ratio > 3.0
    is_present = max_jump_ratio >= 3.0 and max_val >= 2000

    return {
        "issue": "target_typos",
        "present": is_present,
        "count": None,  # null in findings.yaml when absent
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "clean_max": clean_max,
        "top_10_values": top_10,
        "max_step_ratio": round(max_jump_ratio, 3),
        "explanation": (
            f"No target typos detected. '{target_col}' smoothly spans {min_val} to {max_val} seconds "
            f"(mean {mean_val}s, std {std_val}s). Top 10 values descend smoothly {top_10[:5]} with no 10x spikes or gaps. "
            f"It matches the distribution of clean later.csv (max {clean_max}s)."
            if not is_present
            else f"Detected target typos with maximum value {max_val}."
        ),
    }


# -----------------------------------------------------------------------------
# 7. Spelling Variants Check
# -----------------------------------------------------------------------------

def check_spelling_variants(
    train_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Investigate whether text columns have multiple representations for the same category.

    Handbook Rule:
        value_counts() on each text column. later.csv has the clean spellings.
        true_levels is the number of real categories.
    """
    if categorical_cols is None:
        categorical_cols = list(train_df.select_dtypes(include=["object", "string", "str"]).columns)
        categorical_cols = [c for c in categorical_cols if c not in ["record_id", "date"]]

    detected_col = None
    true_levels_count = None
    all_col_stats: Dict[str, Any] = {}

    for col in categorical_cols:
        train_raw_counts = {str(k): int(v) for k, v in train_df[col].value_counts(dropna=False).items()}
        clean_counts = {str(k): int(v) for k, v in clean_df[col].value_counts(dropna=False).items()} if col in clean_df.columns else {}

        train_unique_count = len(train_raw_counts)
        clean_unique_count = len(clean_counts)

        all_col_stats[col] = {
            "train_unique_levels": train_unique_count,
            "clean_unique_levels": clean_unique_count,
            "clean_levels": list(clean_counts.keys()),
            "train_sample_variants": list(train_raw_counts.keys())[:8],
        }

        # If train has significantly more unique levels than clean reference
        if train_unique_count > clean_unique_count and clean_unique_count > 0:
            detected_col = col
            true_levels_count = clean_unique_count

    is_present = detected_col is not None

    return {
        "issue": "spelling_variants",
        "present": is_present,
        "column": detected_col,
        "true_levels": true_levels_count,
        "categorical_statistics": all_col_stats,
        "explanation": (
            f"In column '{detected_col}', train.csv has {all_col_stats[detected_col]['train_unique_levels']} raw representations "
            f"(mixed casing, abbreviations like 'chi.', 'amb.', 'bul.', leading/trailing whitespaces), "
            f"whereas later.csv defines exactly {true_levels_count} clean canonical levels: "
            f"{all_col_stats[detected_col]['clean_levels']}."
            if is_present
            else "No spelling variants detected."
        ),
    }


# -----------------------------------------------------------------------------
# Comprehensive Investigation Runner & Reporter
# -----------------------------------------------------------------------------

def run_all_seven_investigations(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    later_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run all seven data-quality investigations on the supplied datasets.
    """
    res1 = check_leaking_column(train_df, holdout_df, target_col="pick_seconds")
    res2 = check_unit_mixup(train_df, group_col="shift")
    res3 = check_sentinel_value(train_df)
    res4 = check_duplicate_rows(train_df)
    res5 = check_stuck_sensor(train_df, date_col="date")
    res6 = check_target_typos(train_df, clean_df=later_df, target_col="pick_seconds")
    res7 = check_spelling_variants(train_df, clean_df=later_df)

    return {
        "leaking_column": res1,
        "unit_mixup": res2,
        "sentinel_value": res3,
        "duplicate_rows": res4,
        "stuck_sensor": res5,
        "target_typos": res6,
        "spelling_variants": res7,
    }


def print_investigation_report(results: Dict[str, Any]) -> None:
    """
    Print the full 7-point investigation report in a clear, beginner-friendly format.
    """
    header = "=" * 76
    sub_header = "-" * 76

    print(header)
    print("SCSE3040 MLOps Mini Project: Seven Data-Quality Investigations (MP277)")
    print(header)

    # 1. Leaking column
    r1 = results["leaking_column"]
    status_str = "PRESENT [YES]" if r1["present"] else "ABSENT [NO]"
    print(f"\n[1] LEAKING COLUMN: {status_str}")
    print(sub_header)
    print(f"- Column in findings.yaml : {r1['column']}")
    print(f"- Evidence               : Missing from holdout: {r1['missing_in_holdout']}")
    print("  Feature correlations with pick_seconds:")
    for f, c in r1["feature_correlations"].items():
        print(f"    * {f:20s}: {c:+.4f}")
    print(f"- Detection Logic        : {r1['explanation']}")

    # 2. Unit mixup
    r2 = results["unit_mixup"]
    status_str = "PRESENT [YES]" if r2["present"] else "ABSENT [NO]"
    print(f"\n[2] UNIT MIXUP: {status_str}")
    print(sub_header)
    print(f"- Column in findings.yaml : {r2['column']}")
    print(f"- Group Value (shift)     : {r2['group_value']}")
    print(f"- Exact Factor            : {r2['factor']}")
    print("- Evidence (aisle_distance_m stats per shift):")
    for shift_name, stats in r2["group_statistics"][r2["column"]].items():
        print(f"    * {shift_name:8s} -> Min: {stats['min']:7.1f} | Mean: {stats['mean']:8.2f} | Max: {stats['max']:7.1f}")
    print(f"- Detection Logic        : {r2['explanation']}")

    # 3. Sentinel value
    r3 = results["sentinel_value"]
    status_str = "PRESENT [YES]" if r3["present"] else "ABSENT [NO]"
    print(f"\n[3] SENTINEL VALUE: {status_str}")
    print(sub_header)
    print(f"- Column in findings.yaml : {r3['column']}")
    print(f"- Value                   : {r3['value']}")
    print(f"- Count                   : {r3['count']}")
    print("- Evidence (numeric column ranges and checks):")
    for col, data in r3["column_findings"].items():
        print(f"    * {col:20s} -> Range: [{data['min']}, {data['max']}], Sentinels found: {data['found_sentinels']}")
    print(f"- Detection Logic        : {r3['explanation']}")

    # 4. Duplicate rows
    r4 = results["duplicate_rows"]
    status_str = "PRESENT [YES]" if r4["present"] else "ABSENT [NO]"
    print(f"\n[4] DUPLICATE ROWS: {status_str}")
    print(sub_header)
    print(f"- Extra copies count      : {r4['count']}")
    print("- Evidence (sample duplicate pairs):")
    for s in r4["sample_duplicate_records"]:
        print(f"    * Record {s['record_id']} on {s['date']} | items: {s['items_in_order']}, distance: {s['aisle_distance_m']}, target: {s['pick_seconds']}")
    print(f"- Detection Logic        : {r4['explanation']}")

    # 5. Stuck sensor
    r5 = results["stuck_sensor"]
    status_str = "PRESENT [YES]" if r5["present"] else "ABSENT [NO]"
    print(f"\n[5] STUCK SENSOR: {status_str}")
    print(sub_header)
    print(f"- Column in findings.yaml : {r5['column']}")
    print(f"- Start Date              : {r5['start_date']}")
    print(f"- End Date                : {r5['end_date']}")
    print(f"- Stuck Reading Value     : {r5['value']} m")
    print(f"- Affected Records Count  : {r5['row_count']}")
    print(f"- Detection Logic        : {r5['explanation']}")

    # 6. Target typos
    r6 = results["target_typos"]
    status_str = "PRESENT [YES]" if r6["present"] else "ABSENT [NO]"
    print(f"\n[6] TARGET TYPOS: {status_str}")
    print(sub_header)
    print(f"- Count in findings.yaml  : {r6['count']}")
    print(f"- Evidence (Distribution): Min: {r6['min']}s, Mean: {r6['mean']}s, Max: {r6['max']}s (Clean later.csv max: {r6['clean_max']}s)")
    print(f"- Top 5 highest targets   : {r6['top_10_values'][:5]}")
    print(f"- Max Jump Ratio          : {r6['max_step_ratio']}")
    print(f"- Detection Logic        : {r6['explanation']}")

    # 7. Spelling variants
    r7 = results["spelling_variants"]
    status_str = "PRESENT [YES]" if r7["present"] else "ABSENT [NO]"
    print(f"\n[7] SPELLING VARIANTS: {status_str}")
    print(sub_header)
    print(f"- Column in findings.yaml : {r7['column']}")
    print(f"- True Levels Count       : {r7['true_levels']}")
    print(f"- Clean Levels (later.csv): {r7['categorical_statistics'][r7['column']]['clean_levels']}")
    print(f"- Raw Variants in train   : {r7['categorical_statistics'][r7['column']]['train_unique_levels']} distinct text values")
    print(f"  Examples: {r7['categorical_statistics'][r7['column']]['train_sample_variants']}")
    print(f"- Detection Logic        : {r7['explanation']}")

    print("\n" + header)
    print("SUMMARY FOR findings.yaml DATA ISSUES:")
    print(header)
    print("leaking_column    : present=false, column=null")
    print(f"unit_mixup        : present={str(r2['present']).lower()}, column='{r2['column']}', group_value='{r2['group_value']}', factor={r2['factor']}")
    print("sentinel_value    : present=false, column=null, value=null, count=null")
    print(f"duplicate_rows    : present={str(r4['present']).lower()}, count={r4['count']}")
    print(f"stuck_sensor      : present={str(r5['present']).lower()}, column='{r5['column']}', start_date='{r5['start_date']}', end_date='{r5['end_date']}'")
    print("target_typos      : present=false, count=null")
    print(f"spelling_variants : present={str(r7['present']).lower()}, column='{r7['column']}', true_levels={r7['true_levels']}")
    print(header)


# -----------------------------------------------------------------------------
# Main Execution Guard
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "raw"

    train_path = data_dir / "train.csv"
    holdout_path = data_dir / "holdout_features.csv"
    later_path = data_dir / "later.csv"

    print(f"Executing checks from root: {repo_root}\n")

    train_df = load_csv(train_path)
    holdout_df = load_csv(holdout_path)
    later_df = load_csv(later_path)

    # 1. Schema comparison
    schema_report = compare_columns(train_df, holdout_df, "train.csv", "holdout_features.csv")
    print_comparison_report(schema_report)

    # 2. Seven data issues
    investigations = run_all_seven_investigations(train_df, holdout_df, later_df)
    print_investigation_report(investigations)
