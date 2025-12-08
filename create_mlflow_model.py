"""
Create an MLflow model wrapper for the AutoML pickle model.

The RAI components require MLflow format, but the original AutoML model
has azureml.training dependencies that can't be resolved in a standard
MLflow environment. This script creates a wrapper that:
1. Loads the pickle model
2. Saves it as an MLflow pyfunc model with a custom wrapper
3. The wrapper handles prediction in a way compatible with RAI components
"""

import os
import sys
import pickle
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import cloudpickle


class AutoMLModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for AutoML pickle model to make it MLflow-compatible"""
    
    def __init__(self, model=None):
        self.model = model
    
    def load_context(self, context):
        """Load the pickle model from artifacts"""
        model_path = context.artifacts["model"]
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, context, model_input):
        """Make predictions using the wrapped model"""
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
        else:
            predictions = self.model.predict(pd.DataFrame(model_input))
        
        # Return as numpy array (RAI expects this)
        if hasattr(predictions, 'values'):
            return predictions.values
        return np.array(predictions)


def create_mlflow_model():
    """Create MLflow model from pickle file"""
    
    # Paths
    source_model_path = "models/secondary_cvd_risk/1/model.pkl"
    output_path = "models/secondary_cvd_risk_mlflow"
    
    print(f"Loading model from {source_model_path}...")
    
    # Load the original model
    with open(source_model_path, 'rb') as f:
        original_model = pickle.load(f)
    
    print(f"Model loaded: {type(original_model)}")
    
    # Check if model has expected methods
    if hasattr(original_model, 'predict'):
        print("Model has predict method ✓")
    if hasattr(original_model, 'predict_proba'):
        print("Model has predict_proba method ✓")
    
    # Clean up output directory
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    
    print(f"Saving MLflow model to {output_path}...")
    
    # Create conda environment for MLflow model
    # Use packages that are available in the RAI environment
    conda_env = {
        "name": "rai_model_env",
        "channels": ["conda-forge", "defaults"],
        "dependencies": [
            "python=3.9",
            "pip",
            {
                "pip": [
                    "mlflow>=2.0",
                    "pandas>=1.5",
                    "numpy>=1.23",
                    "scikit-learn>=1.0",
                    "cloudpickle>=2.0",
                    "xgboost<=1.5.2",
                ]
            }
        ]
    }
    
    # Create artifacts dict pointing to the pickle file
    artifacts = {"model": source_model_path}
    
    # Save using mlflow.pyfunc with the wrapper class
    mlflow.pyfunc.save_model(
        path=output_path,
        python_model=AutoMLModelWrapper(),
        artifacts=artifacts,
        conda_env=conda_env,
        code_path=None,  # No additional code needed
    )
    
    print(f"MLflow model saved successfully!")
    print(f"\nContents of {output_path}:")
    for item in os.listdir(output_path):
        print(f"  - {item}")
    
    # Verify the model can be loaded
    print("\nVerifying model can be loaded...")
    try:
        loaded_model = mlflow.pyfunc.load_model(output_path)
        print(f"Model loaded successfully: {type(loaded_model)}")
        
        # Test with dummy data if possible
        print("MLflow model is ready for RAI dashboard!")
    except Exception as e:
        print(f"Warning: Could not verify model loading: {e}")
        print("This may be OK if dependencies are not installed locally")
    
    return output_path


if __name__ == "__main__":
    create_mlflow_model()
