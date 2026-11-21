"""
Machine learning preprocessing pipeline and model construction for Problem MP277
(Warehouse Picking Time - Surat Fulfilment Centre).

This module defines:
- The standard feature set (6 numeric, 2 categorical) per PROBLEM.md §5.
- A ColumnTransformer wrapping numeric imputation + scaling and categorical imputation + one-hot encoding.
- RawFeatureCleaner to safely handle any raw variants during inference.
- Model pipeline factory supporting Ridge, RandomForestRegressor, and GradientBoostingRegressor.
"""

from typing import Any, Dict, List, Optional
import sys
from pathlib import Path
import pandas as pd

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mp277.clean import CANONICAL_ZONE_TYPES, clean_unit_mixup, clean_zone_type

# Exact 8 input features required by POST /predict (PROBLEM.md §5)
NUMERIC_FEATURES: List[str] = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
]

CATEGORICAL_FEATURES: List[str] = [
    "zone_type",
    "shift",
]

FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "pick_seconds"


class RawFeatureCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies lightweight cleaning
    (unit mixup conversion and categorical spelling normalization) to incoming
    DataFrames during inference.
    """

    def __init__(self, clean_units: bool = True, clean_categories: bool = True):
        self.clean_units = clean_units
        self.clean_categories = clean_categories

    def fit(self, X: Any, y: Optional[Any] = None) -> "RawFeatureCleaner":
        return self

    def transform(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()
            if self.clean_units and "shift" in X_clean.columns and "aisle_distance_m" in X_clean.columns:
                X_clean = clean_unit_mixup(X_clean)
            if self.clean_categories and "zone_type" in X_clean.columns:
                X_clean = clean_zone_type(X_clean)
            return X_clean
        return X


def build_preprocessor() -> ColumnTransformer:
    """
    Build the scikit-learn ColumnTransformer for numeric and categorical features.

    - Numeric: Median imputation followed by StandardScaler.
    - Categorical: Most frequent imputation followed by OneHotEncoder(handle_unknown='ignore').
    - Remainder='drop' ensures identifiers like record_id and date are excluded.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model_pipeline(model_config: Dict[str, Any], random_state: int = 42) -> Pipeline:
    """
    Factory creating a full scikit-learn Pipeline incorporating raw feature cleaning,
    preprocessing, and the specified regressor.

    Supported model types:
        - 'Ridge'
        - 'RandomForestRegressor'
        - 'GradientBoostingRegressor'

    Parameters:
        model_config: Dictionary containing 'model_type' and model hyperparameters.
        random_state: Seed for reproducible initialization.

    Returns:
        Pipeline: End-to-end trainable and serializable pipeline.
    """
    model_type = model_config.get("model_type")

    if model_type == "Ridge":
        alpha = float(model_config.get("alpha", 1.0))
        regressor = Ridge(alpha=alpha, random_state=random_state)
    elif model_type == "RandomForestRegressor":
        n_estimators = int(model_config.get("n_estimators", 100))
        max_depth = model_config.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    elif model_type == "GradientBoostingRegressor":
        n_estimators = int(model_config.get("n_estimators", 100))
        learning_rate = float(model_config.get("learning_rate", 0.1))
        max_depth = int(model_config.get("max_depth", 3))
        regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type in config: '{model_type}'")

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("cleaner", RawFeatureCleaner()),
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return pipeline
