# Knuckle Joint ML Prototype — Starter Package

## What this package does
- Trains and compares Random Forest, Extra Trees, and HistGradientBoosting models.
- Reports R², MAE, and RMSE on a held-out 20% test split.
- Saves the selected model and metrics.
- Provides a Streamlit UI to enter dimensions/load and view predictions.

## Important data limitation
The dataset targets are synthetic analytical proxies, NOT actual ANSYS/FEA results.
Model metrics describe prediction of these synthetic proxy targets only. Do not claim
they prove FEA accuracy, and do not use the app for engineering safety decisions.

## Files
- `knuckle_joint_synthetic_dataset_1000.csv`: 1,000-row synthetic prototype dataset
- `train_knuckle_ml.py`: model training and evaluation
- `app.py`: Streamlit prediction interface
- `requirements.txt`: Python dependencies
- `ml_outputs/`: created when the training script runs

## Run on Windows
1. Install Python 3.10 or newer.
2. Put all files in the same folder.
3. Open PowerShell in that folder.
4. Create and activate a virtual environment (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
5. Install packages:
   ```powershell
   pip install -r requirements.txt
   ```
6. Train and evaluate:
   ```powershell
   python train_knuckle_ml.py
   ```
7. Launch the app:
   ```powershell
   streamlit run app.py
   ```

The app requires `ml_outputs/best_synthetic_proxy_model.joblib`, so run training first.
