"""
Knuckle Joint ML Prototype UI
Targets are synthetic analytical proxies, NOT ANSYS FEA results.
Run: streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

HERE = Path(__file__).parent
MODEL_PATH = HERE / "ml_outputs" / "best_synthetic_proxy_model.joblib"
META_PATH = HERE / "ml_outputs" / "model_metadata.json"

st.set_page_config(page_title="Knuckle Joint ML Prototype", page_icon="", layout="wide")
st.title("Knuckle Joint — ML Surrogate Prototype")


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.info("Train the model first by running: python train_knuckle_ml.py")
    st.stop()

model = joblib.load(MODEL_PATH)
meta = __import__("json").loads(META_PATH.read_text(encoding="utf-8"))

st.caption(f"Model: {meta['selected_model']} · Dataset: 1,000 rows")

col1, col2, col3 = st.columns(3)
with col1:
    rod = st.slider("Rod diameter (mm)", 20.0, 50.0, 30.0, 0.1)
    pin_ratio = st.slider("Pin / rod diameter ratio", 1.00, 1.20, 1.08, 0.01)
with col2:
    eye_ratio = st.slider("Eye thickness / rod ratio", 1.10, 1.40, 1.25, 0.01)
    fork_ratio = st.slider("Fork thickness / rod ratio", 0.65, 0.85, 0.75, 0.01)
with col3:
    outer_ratio = st.slider("Outer eye diameter / rod ratio", 1.80, 2.20, 2.00, 0.01)
    load = st.slider("Tensile load (kN)", 10.0, 50.0, 30.0, 0.5)

pin = rod * pin_ratio
eye = rod * eye_ratio
fork = rod * fork_ratio
outer = rod * outer_ratio

X = pd.DataFrame([{
    "rod_diameter_mm": rod,
    "pin_diameter_mm": pin,
    "eye_thickness_mm": eye,
    "fork_thickness_mm": fork,
    "outer_eye_diameter_mm": outer,
    "applied_tensile_load_kN": load,
}])

if st.button("Predict", type="primary"):
    pred = model.predict(X)[0]
    stress, deformation, fos = map(float, pred)
    a, b, c = st.columns(3)
    a.metric("Estimated stress proxy", f"{stress:.2f} MPa")
    b.metric("Estimated deformation proxy", f"{deformation:.5f} mm")
    c.metric("Estimated FoS proxy", f"{fos:.2f}")
    st.caption("Predictions are synthetic-proxy estimates, not validated FEA results.")
    st.dataframe(X, use_container_width=True)

st.subheader("Dataset preview")
data = pd.read_csv(HERE / "knuckle_joint_synthetic_dataset_1000.csv")
st.dataframe(data.head(10), use_container_width=True)
