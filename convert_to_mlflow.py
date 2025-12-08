"""
Convert pickle model to MLflow format for RAI dashboard compatibility.
"""

import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path

# Source model path
MODEL_PATH = "models/secondary_cvd_risk/1/model.pkl"
OUTPUT_PATH = "models/secondary_cvd_risk_mlflow"

# Load the pickle model
print(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
print(f"Model: {model}")

# Get feature names from sample data
DATA_PATH = "data/secondary_cvd_risk_min/secondary-cvd-risk.csv"
print(f"\nLoading feature names from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, nrows=1)
feature_names = [col for col in df.columns if col != 'MACE']
print(f"Number of features: {len(feature_names)}")

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Create a signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define input schema based on features
input_schema = Schema([ColSpec("double", name) for name in feature_names])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

print(f"\nSaving MLflow model to: {OUTPUT_PATH}")

# Save as MLflow model
mlflow.sklearn.save_model(
    sk_model=model,
    path=OUTPUT_PATH,
    signature=signature,
    input_example=df[feature_names].iloc[0:1].to_dict(orient='records')[0]
)

print("\n✅ Model converted to MLflow format successfully!")
print(f"\nDirectory contents:")
for item in Path(OUTPUT_PATH).iterdir():
    print(f"  {item.name}")
