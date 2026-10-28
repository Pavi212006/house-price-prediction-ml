"""Dataset, model, and evaluation utilities for the housing demo."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "MedHouseVal"
RANDOM_STATE = 42


def load_housing_data() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Fetch the California Housing dataset and return features, target, metadata."""
    from sklearn.datasets import fetch_california_housing

    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.copy()
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} was not found.")

    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN].astype(float)
    metadata = {
        "dataset_name": "California Housing (scikit-learn)",
        "target_column": TARGET_COLUMN,
        "target_units": "$100,000",
        "feature_names": list(X.columns),
        "feature_summary": {
            name: {
                "min": float(X[name].min()),
                "max": float(X[name].max()),
                "median": float(X[name].median()),
                "description": dataset.feature_names[i],
            }
            for i, name in enumerate(X.columns)
        },
    }
    return X, y, metadata


def build_candidate_models() -> dict[str, Pipeline]:
    """Return comparable pipelines with imputation inside each pipeline."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=160,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute regression metrics, checking input shape and finiteness."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must be non-empty and have equal lengths.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must contain only finite numbers.")

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }
