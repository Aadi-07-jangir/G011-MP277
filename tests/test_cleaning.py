"""
Unit tests for data cleaning module (src/mp277/clean.py).

All tests use in-memory synthetic DataFrames to ensure total isolation
from the external environment and avoid modifying raw CSV files.
"""

import numpy as np
import pandas as pd
import pytest

from src.mp277.clean import (
    CANONICAL_ZONE_TYPES,
    clean_dataset,
    clean_duplicates,
    clean_stuck_sensor,
    clean_unit_mixup,
    clean_zone_type,
)


@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Fixture providing a small synthetic dataset mirroring train.csv schema with known issues."""
    return pd.DataFrame({
        "record_id": ["R001", "R002", "R003", "R004", "R005", "R006", "R006"],
        "date": [
            "2026-01-10",
            "2026-02-15",
            "2026-04-10",  # inside stuck window
            "2026-04-20",  # inside stuck window
            "2026-05-01",  # outside stuck window
            "2026-05-02",
            "2026-05-02",  # duplicate row
        ],
        "items_in_order": [10, 25, 30, 15, 20, 12, 12],
        "aisle_distance_m": [150.0, 25000.0, 120.0, 31000.0, 200.0, 180.0, 180.0],
        "picker_exp_months": [12, 24, 6, 36, 18, 48, 48],
        "shelf_height_m": [1.50, 1.92, 1.92, 1.92, 1.92, 2.10, 2.10],
        "shift_hour": [8, 22, 14, 23, 10, 16, 16],
        "order_weight_kg": [5.2, 14.1, 8.5, 12.0, 7.3, 11.2, 11.2],
        "zone_type": [" ambient ", "CHILLED", "chi.", "bulky ", "amb.", "bul.", "bul."],
        "shift": ["Morning", "Night", "Evening", "Night", "Morning", "Evening", "Evening"],
        "pick_seconds": [210, 420, 315, 380, 250, 290, 290],
    })


def test_clean_unit_mixup():
    """Verify that Night shift distances in cm (>=1000) are converted to metres, leaving others intact."""
    df = pd.DataFrame({
        "shift": ["Night", "Night", "Morning", "Evening"],
        "aisle_distance_m": [25000.0, 180.0, 150.0, 350.0],
    })
    cleaned = clean_unit_mixup(df)

    # Night row with 25,000 cm should convert to 250.0 m
    assert cleaned.loc[0, "aisle_distance_m"] == 250.0
    # Night row with 180.0 m should remain 180.0 m
    assert cleaned.loc[1, "aisle_distance_m"] == 180.0
    # Morning and Evening should remain unchanged
    assert cleaned.loc[2, "aisle_distance_m"] == 150.0
    assert cleaned.loc[3, "aisle_distance_m"] == 350.0


def test_clean_duplicates():
    """Verify that exact duplicate rows are removed, preserving unique rows."""
    df = pd.DataFrame({
        "record_id": ["A", "B", "B", "C"],
        "val": [1, 2, 2, 3],
    })
    cleaned = clean_duplicates(df)
    assert len(cleaned) == 3
    assert list(cleaned["record_id"]) == ["A", "B", "C"]


def test_clean_stuck_sensor():
    """Verify that shelf_height_m == 1.92 is set to NaN only within the confirmed date window."""
    df = pd.DataFrame({
        "date": ["2026-03-30", "2026-04-03", "2026-04-15", "2026-04-25", "2026-04-26"],
        "shelf_height_m": [1.92, 1.92, 1.92, 1.92, 1.92],
    })
    cleaned = clean_stuck_sensor(df)

    # 2026-03-30 is before window -> should keep 1.92
    assert cleaned.loc[0, "shelf_height_m"] == 1.92
    # 2026-04-03 to 2026-04-25 are inside window -> should be NaN
    assert np.isnan(cleaned.loc[1, "shelf_height_m"])
    assert np.isnan(cleaned.loc[2, "shelf_height_m"])
    assert np.isnan(cleaned.loc[3, "shelf_height_m"])
    # 2026-04-26 is after window -> should keep 1.92
    assert cleaned.loc[4, "shelf_height_m"] == 1.92


def test_clean_stuck_sensor_other_values_preserved():
    """Verify that non-1.92 readings inside the stuck window are preserved."""
    df = pd.DataFrame({
        "date": ["2026-04-10"],
        "shelf_height_m": [2.35],
    })
    cleaned = clean_stuck_sensor(df)
    assert cleaned.loc[0, "shelf_height_m"] == 2.35


def test_clean_zone_type():
    """Verify that all raw variations map to canonical categories 'ambient', 'chilled', 'bulky'."""
    raw_variants = [
        "ambient", " chilled", "Chilled", "CHILLED", "chi.",
        "bulky", " bulky", "Bulky", "BULKY", "bul.",
        " ambient", "amb.", "ambient ", "Ambient"
    ]
    df = pd.DataFrame({"zone_type": raw_variants})
    cleaned = clean_zone_type(df)

    unique_categories = set(cleaned["zone_type"].unique())
    assert unique_categories.issubset(set(CANONICAL_ZONE_TYPES))
    assert len(unique_categories) == 3


def test_clean_zone_type_handles_nan_and_unexpected():
    """Verify that clean_zone_type handles NaN and unexpected categories without crashing."""
    df = pd.DataFrame({"zone_type": [np.nan, "frozen_food"]})
    cleaned = clean_zone_type(df)
    assert pd.isna(cleaned.loc[0, "zone_type"])
    assert cleaned.loc[1, "zone_type"] == "frozen_food"


def test_clean_dataset_preserves_columns_and_target(synthetic_sample_df):
    """Verify that all original columns, including the target 'pick_seconds', are preserved."""
    initial_cols = list(synthetic_sample_df.columns)
    cleaned = clean_dataset(synthetic_sample_df)

    # Columns must be identical
    assert list(cleaned.columns) == initial_cols
    # Target column must be intact with correct values
    assert "pick_seconds" in cleaned.columns
    assert cleaned["pick_seconds"].notna().all()


def test_clean_dataset_idempotency(synthetic_sample_df):
    """Verify that applying cleaning twice produces the exact same DataFrame."""
    cleaned_once = clean_dataset(synthetic_sample_df)
    cleaned_twice = clean_dataset(cleaned_once)

    pd.testing.assert_frame_equal(cleaned_once, cleaned_twice)


def test_clean_dataset_without_target_column(synthetic_sample_df):
    """Verify that clean_dataset works cleanly when the target column 'pick_seconds' is absent."""
    df_no_target = synthetic_sample_df.drop(columns=["pick_seconds"])
    cleaned = clean_dataset(df_no_target)

    assert "pick_seconds" not in cleaned.columns
    assert len(cleaned) == 6  # one duplicate removed from 7 rows
    assert cleaned.loc[1, "aisle_distance_m"] == 250.0  # Night shift unit mixup fixed
    assert np.isnan(cleaned.loc[2, "shelf_height_m"])  # stuck sensor fixed
    assert cleaned.loc[0, "zone_type"] == "ambient"  # spelling variant fixed


def test_clean_dataset_does_not_mutate_input(synthetic_sample_df):
    """Verify that clean_dataset does not mutate the original caller DataFrame in-place."""
    original_copy = synthetic_sample_df.copy(deep=True)
    _ = clean_dataset(synthetic_sample_df)

    pd.testing.assert_frame_equal(synthetic_sample_df, original_copy)
