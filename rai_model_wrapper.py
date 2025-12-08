"""
Responsible AI Model Wrapper for Azure ML Dashboard

This module provides a custom model wrapper for AutoML models that is compatible 
with Azure ML Responsible AI dashboard. The wrapper includes:
- A ModelWrapper class with __init__ and predict methods
- Conda environment configuration
- Model registration using Azure ML SDK v2

Based on Microsoft Learn: 
https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/10/Create%20Responsible%20AI%20dashboard.ipynb
https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-mlflow-models
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# Custom Model Wrapper Class for Responsible AI Dashboard
# =============================================================================

class ModelWrapper:
    """
    Custom model wrapper class for Azure ML Responsible AI dashboard.
    
    This wrapper enables custom prediction behavior and is compatible with
    MLflow's pyfunc model format. It wraps an AutoML model to provide:
    - Standard __init__ and predict interface
    - Support for predict_proba (classification) or predict (regression)
    - Compatibility with Responsible AI components
    
    Attributes:
        model: The underlying trained model (AutoML or custom sklearn model)
        model_type: Type of model ('classification' or 'regression')
        feature_names: List of feature names expected by the model
    """
    
    def __init__(self, model=None, model_type='classification', feature_names=None):
        """
        Initialize the model wrapper.
        
        Args:
            model: The trained model object (sklearn-compatible)
            model_type: 'classification' or 'regression'
            feature_names: List of feature column names (optional)
        """
        self._model = model
        self._model_type = model_type
        self._feature_names = feature_names
    
    def predict(self, data):
        """
        Make predictions using the wrapped model.
        
        Args:
            data: Input data (pandas DataFrame or numpy array)
            
        Returns:
            numpy array of predictions
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame if needed for feature name consistency
        if isinstance(data, pd.DataFrame):
            X = data
        elif isinstance(data, np.ndarray):
            if self._feature_names:
                X = pd.DataFrame(data, columns=self._feature_names)
            else:
                X = data
        else:
            X = data
        
        return self._model.predict(X)
    
    def predict_proba(self, data):
        """
        Get prediction probabilities for classification models.
        
        Args:
            data: Input data (pandas DataFrame or numpy array)
            
        Returns:
            numpy array of prediction probabilities
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self._model_type != 'classification':
            raise ValueError("predict_proba is only available for classification models")
        
        # Convert to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            X = data
        elif isinstance(data, np.ndarray):
            if self._feature_names:
                X = pd.DataFrame(data, columns=self._feature_names)
            else:
                X = data
        else:
            X = data
        
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self._model.predict(X)
            # Return as probability-like format
            return np.column_stack([1 - predictions, predictions])
    
    def load_model(self, model_path):
        """
        Load a model from a pickle file.
        
        Args:
            model_path: Path to the saved model file
        """
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
    
    def save_model(self, model_path):
        """
        Save the wrapped model to a pickle file.
        
        Args:
            model_path: Path to save the model
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self._model, f)
    
    @property
    def model(self):
        """Get the underlying model."""
        return self._model
    
    @model.setter
    def model(self, value):
        """Set the underlying model."""
        self._model = value


# =============================================================================
# MLflow PythonModel Wrapper for RAI Dashboard
# =============================================================================

try:
    from mlflow.pyfunc import PythonModel, PythonModelContext
    
    class RAIPythonModelWrapper(PythonModel):
        """
        MLflow PythonModel wrapper for Responsible AI dashboard compatibility.
        
        This wrapper is designed to work with Azure ML's RAI components and
        follows the pattern from Microsoft's official documentation.
        """
        
        def __init__(self, model=None):
            """
            Initialize the wrapper with an optional model.
            
            Args:
                model: Pre-trained sklearn-compatible model
            """
            self._model = model
        
        def load_context(self, context: PythonModelContext):
            """
            Load model artifacts from the MLflow context.
            
            Args:
                context: MLflow PythonModelContext with artifact paths
            """
            import pickle
            
            # Load the main model
            if "model" in context.artifacts:
                with open(context.artifacts["model"], 'rb') as f:
                    self._model = pickle.load(f)
            elif "model.pkl" in context.artifacts:
                with open(context.artifacts["model.pkl"], 'rb') as f:
                    self._model = pickle.load(f)
        
        def predict(self, context: PythonModelContext, data):
            """
            Make predictions using the loaded model.
            
            For classification models, returns prediction probabilities
            to support Responsible AI dashboard features.
            
            Args:
                context: MLflow PythonModelContext
                data: Input data (pandas DataFrame or numpy array)
                
            Returns:
                Model predictions or probabilities
            """
            if hasattr(self._model, 'predict_proba'):
                # Return probabilities for classification
                return self._model.predict_proba(data)
            else:
                # Return predictions for regression
                return self._model.predict(data)

except ImportError:
    # MLflow not installed, skip PythonModel wrapper
    RAIPythonModelWrapper = None


# =============================================================================
# Conda Environment Configuration
# =============================================================================

def get_conda_env_dict():
    """
    Get conda environment specification as a dictionary.
    
    Returns:
        dict: Conda environment specification compatible with MLflow
    """
    return {
        "name": "rai_model_env",
        "channels": ["conda-forge", "defaults"],
        "dependencies": [
            "python=3.9",
            "pip",
            {
                "pip": [
                    "azure-ai-ml>=1.11.0",
                    "azure-identity>=1.12.0",
                    "scikit-learn>=1.0.0",
                    "pandas>=1.5.0",
                    "numpy>=1.23.0",
                    "mlflow>=2.0.0",
                    "shap>=0.42.0",
                    "interpret>=0.4.0",
                    "raiwidgets>=0.25.0",
                    "responsibleai>=0.25.0",
                ]
            }
        ]
    }


def save_conda_env(output_path="conda_env_rai.yml"):
    """
    Save conda environment specification to a YAML file.
    
    Args:
        output_path: Path to save the conda environment file
    """
    import yaml
    
    conda_env = get_conda_env_dict()
    
    with open(output_path, 'w') as f:
        yaml.dump(conda_env, f, default_flow_style=False)
    
    print(f"Conda environment saved to: {output_path}")
    return output_path


# =============================================================================
# Azure ML Model Registration Functions
# =============================================================================

def register_model_for_rai(
    ml_client,
    model_path,
    model_name,
    model_description="Model for Responsible AI dashboard",
    model_type="CUSTOM_MODEL"
):
    """
    Register a model in Azure ML for use with Responsible AI dashboard.
    
    This function registers the model as a custom_model type which is
    compatible with Azure ML's RAI components.
    
    Args:
        ml_client: Azure ML client (MLClient instance)
        model_path: Path to the model directory or file
        model_name: Name for the registered model
        model_description: Description of the model
        model_type: Type of model asset (CUSTOM_MODEL or MLFLOW_MODEL)
        
    Returns:
        Registered model object
    """
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes
    
    # Determine asset type
    if model_type.upper() == "MLFLOW_MODEL":
        asset_type = AssetTypes.MLFLOW_MODEL
    else:
        asset_type = AssetTypes.CUSTOM_MODEL
    
    # Create model entity
    model = Model(
        path=model_path,
        type=asset_type,
        name=model_name,
        description=model_description,
    )
    
    # Register the model
    registered_model = ml_client.models.create_or_update(model)
    
    print(f"Model registered: {registered_model.name}:{registered_model.version}")
    return registered_model


def save_model_with_wrapper(model, output_dir, model_name="model"):
    """
    Save a model with the RAI wrapper structure.
    
    Creates the following structure:
    output_dir/
        model.pkl
        conda_env.yml
        MLmodel (if using MLflow format)
        
    Args:
        model: Trained sklearn-compatible model
        output_dir: Directory to save model artifacts
        model_name: Base name for the model file
        
    Returns:
        Path to the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model_file = output_path / f"{model_name}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_file}")
    
    # Save conda environment
    conda_file = output_path / "conda_env.yml"
    save_conda_env(str(conda_file))
    
    return str(output_path)


def log_model_with_mlflow(
    model,
    artifact_path="model",
    registered_model_name=None,
    conda_env=None
):
    """
    Log a model to MLflow with RAI-compatible format.
    
    Args:
        model: Trained model
        artifact_path: Path in MLflow artifact store
        registered_model_name: Name to register model (optional)
        conda_env: Conda environment dict (optional, uses default if None)
        
    Returns:
        MLflow model info
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    
    if conda_env is None:
        conda_env = get_conda_env_dict()
    
    # Log the model
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            conda_env=conda_env
        )
        
        print(f"Model logged to: {model_info.model_uri}")
    
    return model_info


def log_pyfunc_model_with_mlflow(
    model,
    artifact_path="model",
    registered_model_name=None,
    conda_env=None,
    artifacts=None
):
    """
    Log a model using MLflow's pyfunc format with the RAI wrapper.
    
    This approach allows for custom predict logic and is recommended
    for complex models or when you need to modify prediction behavior.
    
    Args:
        model: Trained model
        artifact_path: Path in MLflow artifact store
        registered_model_name: Name to register model (optional)
        conda_env: Conda environment dict (optional)
        artifacts: Additional artifacts dict (optional)
        
    Returns:
        MLflow model info
    """
    import mlflow
    import mlflow.pyfunc
    
    if conda_env is None:
        conda_env = get_conda_env_dict()
    
    # Create wrapper instance
    wrapper = RAIPythonModelWrapper(model=model)
    
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=wrapper,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            artifacts=artifacts
        )
        
        print(f"PyFunc model logged to: {model_info.model_uri}")
    
    return model_info


# =============================================================================
# Example Usage Functions
# =============================================================================

def example_create_and_register_rai_model():
    """
    Example demonstrating how to create and register a model for RAI dashboard.
    
    This shows the complete workflow:
    1. Train a model (using sklearn)
    2. Wrap it with ModelWrapper
    3. Save with conda environment
    4. Register in Azure ML
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("Example: Creating and Registering a Model for RAI Dashboard")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train a simple model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Create wrapper
    wrapper = ModelWrapper(
        model=model,
        model_type='classification',
        feature_names=[f'feature_{i}' for i in range(10)]
    )
    
    # Test predictions
    predictions = wrapper.predict(X_test)
    probabilities = wrapper.predict_proba(X_test)
    
    print(f"Test predictions shape: {predictions.shape}")
    print(f"Test probabilities shape: {probabilities.shape}")
    print(f"Sample prediction: {predictions[0]}")
    print(f"Sample probability: {probabilities[0]}")
    
    # Save model
    output_dir = save_model_with_wrapper(model, "./rai_model_output")
    print(f"Model saved to: {output_dir}")
    
    print("\nTo register this model in Azure ML, use:")
    print("```python")
    print("from azure.ai.ml import MLClient")
    print("from azure.identity import DefaultAzureCredential")
    print("")
    print("credential = DefaultAzureCredential()")
    print("ml_client = MLClient.from_config(credential=credential)")
    print("")
    print("from rai_model_wrapper import register_model_for_rai")
    print("registered_model = register_model_for_rai(")
    print("    ml_client=ml_client,")
    print("    model_path='./rai_model_output',")
    print("    model_name='my-rai-model',")
    print("    model_type='CUSTOM_MODEL'")
    print(")")
    print("```")


def example_create_rai_dashboard_pipeline():
    """
    Example demonstrating how to create a Responsible AI dashboard pipeline.
    
    This shows the pipeline components needed for RAI dashboard.
    """
    print("=" * 60)
    print("Example: Creating RAI Dashboard Pipeline")
    print("=" * 60)
    
    pipeline_code = '''
from azure.ai.ml import MLClient, Input, dsl
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Connect to workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# Get RAI components from registry
registry_name = "azureml"
ml_client_registry = MLClient(
    credential=credential,
    subscription_id=ml_client.subscription_id,
    resource_group_name=ml_client.resource_group_name,
    registry_name=registry_name,
)

# Load RAI components
label = "latest"
rai_constructor = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_constructor", 
    label=label
)
rai_explanation = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_explanation", 
    label=label
)
rai_erroranalysis = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_erroranalysis", 
    label=label
)
rai_gather = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_gather", 
    label=label
)

# Define pipeline
@dsl.pipeline(
    compute="aml-cluster",
    description="RAI insights dashboard",
    experiment_name="RAI_insights"
)
def rai_pipeline(target_column_name, train_data, test_data, model_id):
    # Construct RAI dashboard
    create_rai_job = rai_constructor(
        title="RAI Dashboard",
        task_type="classification",  # or "regression"
        model_info=model_id,
        model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=f"azureml:{model_id}"),
        train_dataset=train_data,
        test_dataset=test_data,
        target_column_name=target_column_name,
    )
    
    # Add error analysis
    error_job = rai_erroranalysis(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    
    # Add explanations
    explain_job = rai_explanation(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    
    # Gather all insights
    gather_job = rai_gather(
        constructor=create_rai_job.outputs.rai_insights_dashboard,
        insight_1=error_job.outputs.error_analysis,
        insight_2=explain_job.outputs.explanation,
    )
    
    return {"dashboard": gather_job.outputs.dashboard}

# Submit pipeline
train_data = Input(type="mltable", path="azureml:train_data:1", mode="download")
test_data = Input(type="mltable", path="azureml:test_data:1", mode="download")

pipeline_job = rai_pipeline(
    target_column_name="target",
    train_data=train_data,
    test_data=test_data,
    model_id="my-rai-model:1"
)

# Submit to Azure ML
returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"Pipeline job: {returned_job.studio_url}")
'''
    print(pipeline_code)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Responsible AI Model Wrapper for Azure ML")
    print("=" * 60)
    print()
    print("This module provides utilities for wrapping models for use with")
    print("Azure ML Responsible AI dashboard.")
    print()
    print("Key Components:")
    print("  - ModelWrapper: Custom wrapper with __init__ and predict methods")
    print("  - RAIPythonModelWrapper: MLflow PythonModel wrapper")
    print("  - register_model_for_rai: Register model in Azure ML")
    print("  - save_model_with_wrapper: Save model with conda environment")
    print()
    print("Running examples...")
    print()
    
    # Run examples
    example_create_and_register_rai_model()
    print()
    example_create_rai_dashboard_pipeline()
