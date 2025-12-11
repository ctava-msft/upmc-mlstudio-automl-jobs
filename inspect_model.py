"""Quick script to inspect the AutoML model structure"""
import pickle
from pathlib import Path

model_path = Path("./models/secondary_cvd_risk/1/model.pkl")

print(f"Loading model from: {model_path}")

# Try to peek at the pickle without fully loading
with open(model_path, 'rb') as f:
    import pickletools
    pickletools.dis(f, annotate=1)
