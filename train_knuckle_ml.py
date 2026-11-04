"""
Knuckle Joint Synthetic ML Prototype
====================================
Trains baseline regressors on synthetic analytical proxy data.

IMPORTANT:
Targets are synthetic analytical proxies, NOT ANSYS/FEA results.
Metrics measure how well models learn these proxies, not FEA accuracy.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

DATA_PATH = Path(__file__).with_name("knuckle_joint_synthetic_dataset_1000.csv")
OUTPUT_DIR = Path(__file__).with_name("ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "rod_diameter_mm",
    "pin_diameter_mm",
    "eye_thickness_mm",
    "fork_thickness_mm",
    "outer_eye_diameter_mm",
    "applied_tensile_load_kN",
]
TARGETS = [
    "estimated_max_stress_MPa_proxy",
    "estimated_deformation_mm_proxy",
    "factor_of_safety_proxy",
]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
                l2_regularization=0.1, random_state=42
            )
        ),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        for j, target in enumerate(TARGETS):
            results.append({
                "model": name,
                "target": target,
                "R2": float(r2_score(y_test.iloc[:, j], pred[:, j])),
                "MAE": float(mean_absolute_error(y_test.iloc[:, j], pred[:, j])),
                "RMSE": rmse(y_test.iloc[:, j], pred[:, j]),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Select the model with the lowest mean normalized RMSE across targets.
    # Normalization prevents the largest-scale target from dominating.
    score_rows = []
    for name, model in fitted.items():
        pred = model.predict(X_test)
        nrmse = []
        for j in range(len(TARGETS)):
            span = float(y_test.iloc[:, j].max() - y_test.iloc[:, j].min())
            span = span if span > 0 else 1.0
            nrmse.append(rmse(y_test.iloc[:, j], pred[:, j]) / span)
        score_rows.append((name, float(np.mean(nrmse))))
    best_name = min(score_rows, key=lambda x: x[1])[0]
    best_model = fitted[best_name]

    # Retrain selected model on all rows for use in the prototype app.
    best_model.fit(X, y)

    import joblib
    joblib.dump(best_model, OUTPUT_DIR / "best_synthetic_proxy_model.joblib")
    with open(OUTPUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": best_name,
            "features": FEATURES,
            "targets": TARGETS,
            "data_type": "SYNTHETIC_ANALYTICAL_PROXY_NOT_FEA",
            "warning": "Metrics assess synthetic proxy prediction only; they do not validate ANSYS/FEA accuracy."
        }, f, indent=2)

    print("\n=== Held-out test metrics (synthetic proxy targets only) ===")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")
    print("\nWARNING: These results do NOT measure accuracy against actual ANSYS FEA.")

if __name__ == "__main__":
    main()
