"""Train, compare, and evaluate regression models for the housing demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(SRC))

from house_price_prediction.modeling import (  # noqa: E402
    RANDOM_STATE,
    build_candidate_models,
    calculate_metrics,
    load_housing_data,
)


def main() -> None:
    X, y, metadata = load_housing_data()

    # Keep the final test set untouched during candidate model selection.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1764705882,  # approximately 15% of the complete dataset
        random_state=RANDOM_STATE,
    )

    candidates = build_candidate_models()
    validation_results: dict[str, dict[str, float]] = {}
    best_name: str | None = None
    best_rmse = float("inf")

    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Rows: {len(X):,} | Features: {X.shape[1]}")
    print("\nValidation comparison (lower RMSE is better):")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        scores = calculate_metrics(y_val, predictions)
        validation_results[name] = scores
        print(
            f"- {name}: MAE={scores['mae']:.4f}, "
            f"RMSE={scores['rmse']:.4f}, R²={scores['r2']:.4f}"
        )
        if scores["rmse"] < best_rmse:
            best_name = name
            best_rmse = scores["rmse"]

    if best_name is None:
        raise RuntimeError("No candidate model could be selected.")

    final_model = candidates[best_name]
    final_model.fit(X_train_val, y_train_val)
    test_metrics = calculate_metrics(y_test, final_model.predict(X_test))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, ARTIFACTS / "model.joblib")

    result = {
        "dataset": metadata["dataset_name"],
        "target": metadata["target_column"],
        "target_units": metadata["target_units"],
        "random_state": RANDOM_STATE,
        "split_proportions_approx": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "selected_model": best_name,
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "feature_names": metadata["feature_names"],
        "feature_summary": metadata["feature_summary"],
        "limitations": [
            "Historical California housing benchmark; not current market data.",
            "Predictions are for educational demonstration, not financial decisions.",
        ],
    }
    with (ARTIFACTS / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"\nSelected model: {best_name}")
    print("Held-out test metrics:")
    print(
        f"MAE={test_metrics['mae']:.4f}, "
        f"RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}"
    )
    print(f"\nSaved model and metrics to: {ARTIFACTS}")
    print("Note: target values use units of $100,000; do not present these metrics as current market accuracy.")


if __name__ == "__main__":
    main()
