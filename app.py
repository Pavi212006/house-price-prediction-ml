"""Streamlit interface for the trained housing regression model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"

st.set_page_config(page_title="House Price Prediction Demo", page_icon="🏠", layout="wide")
st.title("House Price Prediction")
st.caption("A machine-learning portfolio demo using the California Housing benchmark")
st.warning(
    "Educational demo only. The dataset is historical and does not represent current listing prices. "
    "Do not use these outputs for financial or property-purchase decisions."
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.info("The model has not been trained in this folder yet. Run python train_model.py, then restart this app.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

st.subheader("Model evaluation")
col1, col2, col3 = st.columns(3)
test = metrics["test_metrics"]
col1.metric("Test MAE", f"{test['mae']:.3f} × $100,000")
col2.metric("Test RMSE", f"{test['rmse']:.3f} × $100,000")
col3.metric("Test R²", f"{test['r2']:.3f}")
st.caption(f"Selected model: {metrics['selected_model']} · Metrics are from a held-out test set.")

st.subheader("Try a prediction")
st.write("Adjust the input features using values within the benchmark dataset's observed range.")
feature_summary = metrics["feature_summary"]
feature_names = metrics["feature_names"]

with st.form("prediction_form"):
    values: dict[str, float] = {}
    left, right = st.columns(2)
    for i, feature in enumerate(feature_names):
        info = feature_summary[feature]
        target_column = left if i % 2 == 0 else right
        with target_column:
            values[feature] = st.number_input(
                feature,
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["median"]),
                help=f"Observed benchmark range: {info['min']:.3f} to {info['max']:.3f}",
                format="%.3f",
            )
    submitted = st.form_submit_button("Estimate median house value", type="primary")

if submitted:
    input_frame = pd.DataFrame([values], columns=feature_names)
    try:
        prediction_units = float(model.predict(input_frame)[0])
        predicted_dollars = prediction_units * 100_000
        if predicted_dollars <= 0:
            st.warning("The model returned a non-positive estimate. Treat this as an out-of-range model limitation.")
        else:
            st.success(f"Estimated benchmark median value: ${predicted_dollars:,.0f}")
        st.caption("The dataset target is measured in units of $100,000. This estimate is not a current property valuation.")
    except (ValueError, TypeError) as exc:
        st.error(f"Could not generate a prediction from these inputs: {exc}")

with st.expander("What do these features mean?"):
    st.markdown(
        "- **MedInc:** median income for the block group.\n"
        "- **HouseAge:** median house age.\n"
        "- **AveRooms / AveBedrms:** average rooms and bedrooms per household.\n"
        "- **Population / AveOccup:** block-group population and average occupancy.\n"
        "- **Latitude / Longitude:** approximate geographic location.\n\n"
        "Features describe census block groups; they are not attributes of a single listed home."
    )
