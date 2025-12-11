"""Inspect the AutoML LightGBM wrapper to find the underlying model"""
import pickle
from pathlib import Path

# Load the extracted model
model_path = Path("./models/secondary_cvd_risk/sklearn_model_extracted.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Get the LightGBM estimator
lgbm_wrapper = model.steps[-1][1]

print(f"Wrapper type: {type(lgbm_wrapper)}")
print(f"\nWrapper attributes:")
for attr in dir(lgbm_wrapper):
    if not attr.startswith('__'):
        print(f"  - {attr}")

print(f"\nLooking for the underlying LightGBM model...")
candidates = ['_model', 'model_', '_estimator', 'estimator_', 'model', 'estimator', '_lgbm_model', 'booster_']
for attr in candidates:
    if hasattr(lgbm_wrapper, attr):
        val = getattr(lgbm_wrapper, attr)
        print(f"  Found {attr}: {type(val)}")
