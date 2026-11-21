"""
Unit tests for machine learning pipeline construction, preprocessing, and inference
(src/mp277/pipeline.py).

All tests execute in-memory on synthetic DataFrames without touching raw CSV files.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.mp277.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    create_model_pipeline,
)


@pytest.fixture
def synthetic_train_val_dfs():
    """Provides small train and val synthetic data matching the 8 required features + target."""
    rng = np.random.RandomState(42)
    n_train = 50
    n_val = 15

    def make_df(n):
        return pd.DataFrame({
            "record_id": [f"REC_{i}" for i in range(n)],
            "date": ["2026-02-01"] * n,
            "items_in_order": rng.randint(1, 40, size=n),
            "aisle_distance_m": rng.uniform(20.0, 380.0, size=n),
            "picker_exp_months": rng.randint(0, 60, size=n),
            "shelf_height_m": rng.uniform(0.5, 2.5, size=n),
            "shift_hour": rng.randint(0, 24, size=n),
            "order_weight_kg": rng.uniform(1.0, 25.0, size=n),
            "zone_type": rng.choice(["ambient", "chilled", "bulky"], size=n),
            "shift": rng.choice(["Morning", "Evening", "Night"], size=n),
            "pick_seconds": rng.uniform(100.0, 500.0, size=n),
        })

    train_df = make_df(n_train)
    val_df = make_df(n_val)
    return train_df, val_df


def test_build_preprocessor():
    """Verify that the ColumnTransformer is correctly configured for numeric and categorical subsets."""
    preprocessor = build_preprocessor()
    assert preprocessor.remainder == "drop"

    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "Ridge", "alpha": 1.0},
        {"model_type": "RandomForestRegressor", "n_estimators": 10, "max_depth": 4},
        {"model_type": "GradientBoostingRegressor", "n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
    ],
)
def test_create_model_pipeline_all_candidates(config, synthetic_train_val_dfs):
    """Verify that all three model configurations in params.yaml instantiate and fit properly."""
    train_df, val_df = synthetic_train_val_dfs
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    pipeline = create_model_pipeline(config, random_state=42)
    assert isinstance(pipeline, Pipeline)

    pipeline.fit(X_train, y_train)

    X_val = val_df[FEATURE_COLUMNS]
    preds = pipeline.predict(X_val)

    assert len(preds) == len(val_df)
    assert np.all(np.isfinite(preds))


def test_pipeline_handles_unknown_categorical_values(synthetic_train_val_dfs):
    """Verify that unseen categorical levels during inference do not crash the pipeline."""
    train_df, _ = synthetic_train_val_dfs
    pipeline = create_model_pipeline({"model_type": "Ridge", "alpha": 1.0})
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    # Unseen categories: 'DeepFreeze' for zone_type, 'Graveyard' for shift
    unseen_df = pd.DataFrame({
        "items_in_order": [15],
        "aisle_distance_m": [150.0],
        "picker_exp_months": [12],
        "shelf_height_m": [1.5],
        "shift_hour": [8],
        "order_weight_kg": [10.0],
        "zone_type": ["DeepFreeze"],
        "shift": ["Graveyard"],
    })

    preds = pipeline.predict(unseen_df)
    assert len(preds) == 1
    assert np.isfinite(preds[0])


def test_pipeline_ignores_extra_columns(synthetic_train_val_dfs):
    """Verify that passing extra metadata columns (record_id, date, extra_col) does not break inference."""
    train_df, val_df = synthetic_train_val_dfs
    pipeline = create_model_pipeline({"model_type": "Ridge", "alpha": 1.0})
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    # Pass val_df with record_id, date, and an unknown column
    val_with_extras = val_df.copy()
    val_with_extras["extra_noise_col"] = 999.0

    preds = pipeline.predict(val_with_extras)
    assert len(preds) == len(val_df)
    assert np.all(np.isfinite(preds))


def test_pipeline_no_data_leakage(synthetic_train_val_dfs):
    """Verify that numeric scaling statistics and imputation medians are learned strictly from train data."""
    train_df, val_df = synthetic_train_val_dfs

    # Ensure shelf_height_m has known distinct medians
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["shelf_height_m"] = 1.0
    val_df["shelf_height_m"] = 10.0  # completely different scale

    pipeline = create_model_pipeline({"model_type": "Ridge", "alpha": 1.0})
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    # Inspect the fitted imputer and scaler in the pipeline
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_num_pipeline = fitted_preprocessor.named_transformers_["num"]
    fitted_imputer = fitted_num_pipeline.named_steps["imputer"]
    fitted_scaler = fitted_num_pipeline.named_steps["scaler"]

    # Imputer statistics should reflect train_df, not val_df
    shelf_idx = NUMERIC_FEATURES.index("shelf_height_m")
    assert fitted_imputer.statistics_[shelf_idx] == 1.0
    assert fitted_scaler.mean_[shelf_idx] == 1.0


def test_pipeline_cleans_raw_unprocessed_inference_features():
    """Verify that raw feature cleaning (raw zone_type variants and Night cm distances) works inside pipeline."""
    # Synthetic training data
    train_df = pd.DataFrame({
        "items_in_order": [10, 20, 30],
        "aisle_distance_m": [100.0, 200.0, 300.0],
        "picker_exp_months": [5, 15, 25],
        "shelf_height_m": [1.2, 1.8, 2.0],
        "shift_hour": [8, 14, 20],
        "order_weight_kg": [5.0, 10.0, 15.0],
        "zone_type": ["ambient", "chilled", "bulky"],
        "shift": ["Morning", "Evening", "Night"],
        "pick_seconds": [150, 250, 350],
    })

    pipeline = create_model_pipeline({"model_type": "Ridge", "alpha": 1.0})
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    # Unprocessed raw inference inputs: ' chi.' instead of 'chilled', 25000 cm instead of 250 m
    raw_input_df = pd.DataFrame({
        "items_in_order": [25],
        "aisle_distance_m": [25000.0],
        "picker_exp_months": [10],
        "shelf_height_m": [1.5],
        "shift_hour": [22],
        "order_weight_kg": [8.0],
        "zone_type": [" chi."],
        "shift": ["Night"],
    })

    preds = pipeline.predict(raw_input_df)
    assert len(preds) == 1
    assert np.isfinite(preds[0])
