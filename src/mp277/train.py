"""
Model training and MLflow experiment tracking module for Problem MP277
(Warehouse Picking Time - Surat Fulfilment Centre).

This module:
1. Loads parameters from params.yaml.
2. Computes the SHA256 hash of the original train.csv.
3. Cleans training data using src.mp277.clean.clean_dataset.
4. Partitions data into training and validation sets.
5. Trains three candidate configurations independently.
6. Logs parameters, metrics (including MAE and data_sha256), and pipelines to MLflow.
7. Selects the champion candidate based on validation MAE.
8. Registers Version 1 in the MLflow Model Registry and sets the @champion alias.
9. Saves the champion pipeline to models/model.joblib.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import sys

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import yaml

from src.mp277.clean import clean_dataset
from src.mp277.pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_model_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Calculate the SHA256 hash of raw file bytes."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found for SHA256 computation: {path.resolve()}")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_config(params_path: Union[str, Path] = "params.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_training_data(
    train_path: Union[str, Path],
    val_size: float = 0.2,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load raw training data, apply cleaning, and split into train/val partitions.

    Returns:
        X_train, X_val, y_train, y_val
    """
    path = Path(train_path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path.resolve()}")

    df_raw = pd.read_csv(path)
    logger.info(f"Loaded raw training data from {path.name}: {df_raw.shape}")

    # Clean data (deduplication, unit mix-up, stuck sensor invalidation, spelling variants)
    df_cleaned = clean_dataset(df_raw)
    logger.info(f"Cleaned training data shape: {df_cleaned.shape}")

    if target_column not in df_cleaned.columns:
        raise KeyError(f"Target column '{target_column}' missing from cleaned data.")

    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    logger.info(
        f"Partitioned data: Train={len(X_train)} samples, Validation={len(X_val)} samples."
    )
    return X_train, X_val, y_train, y_val


def run_training(
    params_path: Union[str, Path] = "params.yaml",
    override_tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute full training workflow:
    - Train 3+ candidate models.
    - Track metrics, parameters, and models in MLflow.
    - Register the champion model and set the @champion alias.
    - Export models/model.joblib.

    Parameters:
        params_path: Path to params.yaml.
        override_tracking_uri: Optional custom tracking URI (for testing isolation).

    Returns:
        Dict[str, Any]: Summary of experiment runs, best model metrics, and registration info.
    """
    config = load_config(params_path)

    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    models_cfg = config.get("models", {})
    tracking_cfg = config.get("tracking", {})

    train_path = Path(data_cfg.get("train_path", "data/raw/train.csv"))
    target_column = data_cfg.get("target_column", TARGET_COLUMN)
    val_size = float(split_cfg.get("val_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))

    experiment_name = tracking_cfg.get("experiment_name", "mp277-picking-time")
    registered_model_name = tracking_cfg.get("registered_model_name", "mp277-picking-model")
    tracking_uri = override_tracking_uri or tracking_cfg.get("tracking_uri", "sqlite:///mlflow.db")

    # 1. Compute SHA256 of the raw uncleaned training file
    data_sha256 = compute_file_sha256(train_path)
    logger.info(f"Raw data SHA256: {data_sha256}")

    # 2. Prepare train/val splits
    X_train, X_val, y_train, y_val = prepare_training_data(
        train_path=train_path,
        val_size=val_size,
        random_state=random_state,
        target_column=target_column,
    )

    # 3. Configure MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {tracking_uri}, Experiment: {experiment_name}")

    client = MlflowClient(tracking_uri=tracking_uri)

    candidate_results: List[Dict[str, Any]] = []

    # 4. Train and log each candidate model
    for candidate_key, candidate_cfg in models_cfg.items():
        model_type = candidate_cfg.get("model_type")
        logger.info(f"--- Training candidate: {candidate_key} ({model_type}) ---")

        with mlflow.start_run(run_name=candidate_key) as run:
            run_id = run.info.run_id

            # Instantiate and fit pipeline
            pipeline = create_model_pipeline(candidate_cfg, random_state=random_state)
            pipeline.fit(X_train, y_train)

            # Evaluate predictions
            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            logger.info(f"{candidate_key} Validation MAE: {val_mae:.4f}s (Train MAE: {train_mae:.4f}s)")

            # Log parameters (MANDATORY: data_sha256 on every run)
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("candidate_name", candidate_key)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("random_state", random_state)
            for param_k, param_v in candidate_cfg.items():
                if param_k != "model_type":
                    mlflow.log_param(param_k, param_v)

            # Log metrics (MANDATORY: metric name containing 'mae')
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("mae", val_mae)

            # Log scikit-learn model artifact using MLflow 3 standard
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            candidate_results.append({
                "candidate_key": candidate_key,
                "model_type": model_type,
                "run_id": run_id,
                "val_mae": val_mae,
                "train_mae": train_mae,
                "pipeline": pipeline,
                "model_uri": model_info.model_uri,
            })

    if not candidate_results:
        raise RuntimeError("No candidate models were trained.")

    # 5. Select best candidate based on validation MAE
    best_candidate = min(candidate_results, key=lambda r: r["val_mae"])
    best_run_id = best_candidate["run_id"]
    best_pipeline = best_candidate["pipeline"]
    best_val_mae = best_candidate["val_mae"]
    best_model_uri = best_candidate["model_uri"]

    logger.info(
        f"Selected best candidate: {best_candidate['candidate_key']} "
        f"with Validation MAE = {best_val_mae:.4f}s (Run ID: {best_run_id})"
    )

    # 6. Register model as Version 1
    logger.info(f"Registering model from {best_model_uri} under name '{registered_model_name}'...")
    model_version = mlflow.register_model(model_uri=best_model_uri, name=registered_model_name)
    version_number = str(model_version.version)

    # 7. Set @champion alias
    logger.info(f"Setting alias 'champion' on model '{registered_model_name}' version {version_number}...")
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=version_number,
    )

    # 8. Save selected fitted pipeline artifact to models/model.joblib
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "model.joblib"
    joblib.dump(best_pipeline, joblib_path)
    logger.info(f"Saved champion pipeline to {joblib_path.resolve()}")

    return {
        "data_sha256": data_sha256,
        "runs": [
            {
                "candidate": r["candidate_key"],
                "model_type": r["model_type"],
                "run_id": r["run_id"],
                "val_mae": r["val_mae"],
                "train_mae": r["train_mae"],
            }
            for r in candidate_results
        ],
        "best_candidate": best_candidate["candidate_key"],
        "best_run_id": best_run_id,
        "best_val_mae": best_val_mae,
        "registered_model_name": registered_model_name,
        "registered_version": version_number,
        "champion_alias": "champion",
        "joblib_path": str(joblib_path),
    }


if __name__ == "__main__":
    results = run_training()
    print("\n" + "=" * 60)
    print("TRAINING & MLFLOW REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Data SHA256           : {results['data_sha256']}")
    print(f"Best Candidate        : {results['best_candidate']}")
    print(f"Best Validation MAE   : {results['best_val_mae']:.4f} seconds")
    print(f"Best Run ID           : {results['best_run_id']}")
    print(f"Registered Model Name : {results['registered_model_name']}")
    print(f"Registered Version    : {results['registered_version']}")
    print(f"Model Alias           : @{results['champion_alias']}")
    print(f"Saved Joblib Artifact : {results['joblib_path']}")
    print("=" * 60)
