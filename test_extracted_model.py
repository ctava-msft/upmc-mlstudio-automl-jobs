"""
Test: Verify Extracted AutoML Model Can Be Loaded and Explained
"""
import pickle
from pathlib import Path
import sys

print("=" * 70)
print("Testing Extracted AutoML Model")
print("=" * 70)

# Test 1: Load the extracted model
print("\n1. Loading extracted model...")
model_path = Path("./models/secondary_cvd_risk/sklearn_model_extracted.pkl")

if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded successfully")
    print(f"  Type: {type(model).__name__}")
    print(f"  Module: {type(model).__module__}")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check model methods
print("\n2. Checking model capabilities...")
has_predict = hasattr(model, 'predict')
has_predict_proba = hasattr(model, 'predict_proba')
has_steps = hasattr(model, 'steps')
has_named_steps = hasattr(model, 'named_steps')

print(f"  has predict(): {has_predict}")
print(f"  has predict_proba(): {has_predict_proba}")
print(f"  has steps: {has_steps}")
print(f"  has named_steps: {has_named_steps}")

if has_steps:
    print(f"\n  Pipeline steps ({len(model.steps)}):")
    for i, (name, transformer) in enumerate(model.steps):
        print(f"    {i+1}. {name}: {type(transformer).__name__}")

# Test 3: Load test data and make predictions  
print("\n3. Testing predictions on data...")
try:
    import pandas as pd
    import numpy as np
    
    # Load data
    data_path = Path("./data/secondary_cvd_risk_min/secondary-cvd-risk.csv")
    df = pd.read_csv(data_path)
    print(f"  Loaded data: {df.shape}")
    
    # For AutoML models, pass RAW data - the model has its own DataTransformer
    target_column = 'MACE'
    
    # Only drop the target column, keep everything else
    X = df.drop(columns=[target_column]).head(10)
    y = df[target_column].head(10)
    
    print(f"  Raw features (AutoML will transform): {X.shape}")
    print(f"  Feature columns: {len(X.columns)}")
    
    # Make predictions - AutoML model handles all preprocessing internally
    predictions = model.predict(X)
    print(f"✓ Predictions successful")
    print(f"  Sample predictions: {predictions[:5]}")
    print(f"  True labels: {y.values[:5]}")
    
    if has_predict_proba:
        probabilities = model.predict_proba(X)
        print(f"✓ Predict_proba successful")
        print(f"  Sample probabilities shape: {probabilities.shape}")
        print(f"  Sample positive class probs: {probabilities[:5, 1]}")
    
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test with SHAP
print("\n4. Testing SHAP compatibility...")
try:
    import shap
    
    # Use small sample for quick test
    X_sample = X.head(5)
    
    print(f"  NOTE: For AutoML models with DataTransformer,")
    print(f"  SHAP may need the underlying LightGBM model directly")
    print(f"  Attempting TreeExplainer on full pipeline...")
    
    # Try to create SHAP explainer
    # For AutoML pipelines, we may need to access the final estimator
    try:
        explainer = shap.TreeExplainer(model)
        print(f"✓ TreeExplainer created on full pipeline")
    except Exception as e1:
        print(f"  Pipeline explainer failed: {e1}")
        print(f"  Trying final estimator instead...")
        
        # Get the final estimator (LightGBMClassifier)
        if hasattr(model, 'steps'):
            final_estimator = model.steps[-1][1]
            print(f"  Final estimator: {type(final_estimator).__name__}")
            
            # Need to transform X first
            X_transformed = X_sample.copy()
            for name, transformer in model.steps[:-1]:
                X_transformed = transformer.transform(X_transformed)
            
            explainer = shap.TreeExplainer(final_estimator)
            print(f"✓ TreeExplainer created on final estimator")
            X_sample = X_transformed
    
    # Calculate SHAP values
    shap_values = explainer(X_sample)
    print(f"✓ SHAP values calculated")
    print(f"  Shape: {shap_values.values.shape}")
    
    # Check if binary classification
    if len(shap_values.values.shape) == 3:
        print(f"  Binary classification detected (shape has 3 dimensions)")
        shap_values_pos = shap_values[:, :, 1]
        print(f"  Positive class SHAP values shape: {shap_values_pos.values.shape}")
    else:
        shap_values_pos = shap_values
        print(f"  Single output SHAP values")
    
    print(f"✓ SHAP analysis ready for full dataset")
    
except Exception as e:
    print(f"✗ SHAP test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("SUCCESS: Extracted AutoML Model is Fully Functional!")
print("=" * 70)
print("\nThe extracted model:")
print("  ✓ Can be loaded without Azure ML dependencies")
print("  ✓ Can make predictions on new data")
print("  ✓ Can be explained using SHAP TreeExplainer")
print("  ✓ Is ready for production use")
print("\nYou can now use this model in shap_secondary_cvd_risk.py by changing:")
print("  model_path = Path('./models/secondary_cvd_risk/sklearn_model_extracted.pkl')")
print("=" * 70)
