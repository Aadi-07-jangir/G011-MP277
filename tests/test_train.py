"""
Unit tests for model training, MLflow tracking, and model registration (src/mp277/train.py).

All tests use temporary isolated SQLite tracking databases and synthetic datasets
to ensure zero side effects on the project's production mlflow.db or original raw CSVs.
"""

from pathlib import Path
import hashlib
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
import yaml

from src.mp277.train import compute_file_sha256, run_training


@pytest.fixture
def isolated_training_env(tmp_path: Path):
    """Sets up an isolated directory with synthetic train.csv, params.yaml, and SQLite DB."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "synthetic_train.csv"

    # Generate synthetic training records
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({
        "record_id": [f"R_{i:04d}" for i in range(n)],
        "date": ["2026-02-10"] * n,
        "items_in_order": rng.randint(5, 35, size=n),
        "aisle_distance_m": rng.uniform(50.0, 350.0, size=n),
        "picker_exp_months": rng.randint(1, 48, size=n),
        "shelf_height_m": rng.uniform(0.8, 2.2, size=n),
        "shift_hour": rng.randint(6, 22, size=n),
        "order_weight_kg": rng.uniform(2.0, 20.0, size=n),
        "zone_type": rng.choice(["ambient", "chilled", "bulky"], size=n),
        "shift": rng.choice(["Morning", "Evening", "Night"], size=n),
        "pick_seconds": rng.uniform(120.0, 450.0, size=n),
    })
    df.to_csv(train_file, index=False)

    # Compute expected SHA256 of raw synthetic file
    with open(train_file, "rb") as f:
        expected_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Create temporary params.yaml
    db_file = tmp_path / "isolated_mlflow.db"
    tracking_uri = f"sqlite:///{db_file.as_posix()}"

    params = {
        "data": {
            "train_path": str(train_file),
            "later_path": str(train_file),
            "holdout_path": str(train_file),
            "target_column": "pick_seconds",
        },
        "split": {
            "val_size": 0.25,
            "random_state": 42,
        },
        "models": {
            "candidate_ridge": {"model_type": "Ridge", "alpha": 1.0},
            "candidate_rf": {"model_type": "RandomForestRegressor", "n_estimators": 5, "max_depth": 3, "random_state": 42},
            "candidate_gb": {"model_type": "GradientBoostingRegressor", "n_estimators": 5, "learning_rate": 0.1, "max_depth": 2, "random_state": 42},
        },
        "tracking": {
            "experiment_name": "test-experiment",
            "registered_model_name": "test-picking-model",
            "tracking_uri": tracking_uri,
        },
    }

    params_file = tmp_path / "params.yaml"
    with open(params_file, "w") as f:
        yaml.safe_dump(params, f)

    return {
        "params_path": params_file,
        "tracking_uri": tracking_uri,
        "train_file": train_file,
        "expected_sha256": expected_sha256,
        "registered_model_name": "test-picking-model",
    }


def test_compute_file_sha256(isolated_training_env):
    """Verify that compute_file_sha256 computes exact byte SHA256."""
    train_file = isolated_training_env["train_file"]
    expected_sha256 = isolated_training_env["expected_sha256"]

    computed = compute_file_sha256(train_file)
    assert computed == expected_sha256


def test_run_training_logs_runs_and_registers_champion(isolated_training_env):
    """Verify end-to-end training: logs runs with MAE and data_sha256, registers champion."""
    params_path = isolated_training_env["params_path"]
    tracking_uri = isolated_training_env["tracking_uri"]
    expected_sha256 = isolated_training_env["expected_sha256"]
    reg_name = isolated_training_env["registered_model_name"]

    results = run_training(params_path=params_path, override_tracking_uri=tracking_uri)

    # 1. Verify returned result structure
    assert results["data_sha256"] == expected_sha256
    assert len(results["runs"]) == 3
    assert results["registered_model_name"] == reg_name
    assert results["champion_alias"] == "champion"
    assert results["best_val_mae"] > 0.0

    # 2. Inspect MLflow tracking database directly
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 3

    # Check each run logged data_sha256 and metric containing mae
    for run in runs:
        params_dict = run.data.params
        metrics_dict = run.data.metrics

        assert "data_sha256" in params_dict
        assert params_dict["data_sha256"] == expected_sha256

        # Must log metrics containing 'mae'
        mae_metrics = [k for k in metrics_dict if "mae" in k]
        assert len(mae_metrics) > 0
        assert "val_mae" in metrics_dict
        assert metrics_dict["val_mae"] > 0

    # 3. Verify Model Registry and @champion alias
    registered_model = client.get_registered_model(reg_name)
    assert registered_model.name == reg_name
    assert "champion" in registered_model.aliases
    assert int(registered_model.aliases["champion"]) == 1

    # 4. Verify saved Joblib artifact is valid and can predict
    joblib_path = Path(results["joblib_path"])
    assert joblib_path.exists()
    loaded_pipeline = joblib.load(joblib_path)

    sample_test_df = pd.DataFrame({
        "items_in_order": [20],
        "aisle_distance_m": [150.0],
        "picker_exp_months": [12],
        "shelf_height_m": [1.8],
        "shift_hour": [10],
        "order_weight_kg": [7.5],
        "zone_type": ["ambient"],
        "shift": ["Morning"],
    })
    preds = loaded_pipeline.predict(sample_test_df)
    assert len(preds) == 1
    assert np.isfinite(preds[0])
