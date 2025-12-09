"""
Register RAI-Compatible Model and Run Explanation Job

This script:
1. Creates a sklearn-compatible wrapper around the AutoML model
2. Registers the wrapped model in Azure ML
3. Submits a RAI explanation job

The wrapper makes the AutoML model compatible with RAI components by:
- Implementing predict() and predict_proba() methods
- Providing feature_names_in_ and classes_ attributes
- Being pickleable without AutoML internal dependencies
"""

import os
import sys
import shutil
import pickle
import argparse
import logging
import tempfile
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.entities import Model, Environment
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential

# Load environment variables
load_dotenv()

# Feature names for the secondary CVD risk model
FEATURE_NAMES = [
    'AGE', 'SEX', 'RACE', 'RACE_LABEL', 'ETHNICITY', 'ETHNICITY_DETAILED',
    'ADMITDATE', 'PROC_DATE', 'DISCHARGEDATE', 'HOSPITAL', 'PROC_TYPE',
    'BMI_IP', 'SBP_FIRST', 'SBP_LAST', 'DBP_FIRST', 'DBP_LAST', 'HR_FIRST', 'HR_LAST',
    'CANCER_DX', 'ALZHEIMER_DX', 'NONSPECIFIC_MCI_DX', 'VASCULAR_COGNITIVE_IMPAIRMENT_DX',
    'NONSPECIFIC_COGNITIVE_DEFICIT_DX', 'CHF_HST', 'DIAB_HST', 'AFIB_HST', 'OBESE_HST',
    'MORBIDOBESE_HST', 'TIA_HST', 'CARDIOMYOPATHY_HST', 'TOBACCO_STATUS', 'TOBACCO_STATUS_LABEL',
    'ALCOHOL_STATUS', 'ALCOHOL_STATUS_LABEL', 'ILL_DRUG_STATUS', 'ILL_DRUG_STATUS_LABEL',
    'CCI_CHF', 'CCI_PERIPHERAL_VASC', 'CCI_DEMENTIA', 'CCI_COPD', 'CCI_RHEUMATIC_DISEASE',
    'CCI_PEPTIC_ULCER', 'CCI_MILD_LIVER_DISEASE', 'CCI_DM_NO_CC', 'CCI_DM_WITH_CC',
    'CCI_HEMIPLEGIA', 'CCI_RENAL_DISEASE', 'CCI_MALIG_NO_SKIN', 'CCI_SEVERE_LIVER_DISEASE',
    'CCI_METASTATIC_TUMOR', 'CCI_AIDS_HIV', 'CCI_TOTAL_SCORE',
    'ELIX_CARDIAC_ARRTHYTHMIAS', 'ELIX_CONGESTIVE_HEART_FAILURE', 'ELIX_VALVULAR_DISEASE',
    'ELIX_PULM_CIRC_DISORDERS', 'ELIX_PERIPH_VASC_DISEASE', 'ELIX_HYPERTENSION',
    'ELIX_PARALYSIS', 'ELIX_NEURO_DISORDERS', 'ELIX_COPD', 'ELIX_DIABETES_WO_CC',
    'ELIX_DIABETES_W_CC', 'ELIX_HYPOTHYROIDISM', 'ELIX_RENAL_FAILURE', 'ELIX_LIVER_DISEASE',
    'ELIX_CHRONIC_PEPTIC_ULCER_DISEASE', 'ELIX_HIV_AIDS', 'ELIX_LYMPHOMA',
    'ELIX_METASTATIC_CANCER', 'ELIX_TUMOR_WO_METASTATIC_CANCER', 'ELIX_RHEUMATOID_ARTHRITIS',
    'ELIX_COAGULATION_DEFICIENCY', 'ELIX_OBESITY', 'ELIX_WEIGHT_LOSS',
    'ELIX_FLUID_ELECTROLYTE_DISORDERS', 'ELIX_ANEMIA_BLOOD_LOSS', 'ELIX_DEFICIENCY_ANEMIAS',
    'ELIX_ALCOHOL_ABUSE', 'ELIX_DRUG_ABUSE', 'ELIX_PSYCHOSES', 'ELIX_DEPRESSION',
    'ELIX_AHRQ_SCORE', 'ELIX_VAN_WALRAVEN_SCORE',
    'MED_CURRENT_ASA', 'MED_CURRENT_STATIN', 'MED_CURRENT_LOW_STATIN', 'MED_CURRENT_MODERATE_STATIN',
    'MED_CURRENT_HIGH_STATIN', 'MED_CURRENT_BB', 'MED_CURRENT_AB', 'MED_CURRENT_CCB',
    'MED_CURRENT_ARB', 'MED_CURRENT_ZETIA', 'MED_CURRENT_PCSK9', 'MED_CURRENT_WARFARIN',
    'MED_CURRENT_DOAC', 'MED_CURRENT_COLCHICINE', 'MED_CURRENT_ARNI', 'MED_CURRENT_HYDRALAZINE',
    'MED_CURRENT_MRA', 'MED_CURRENT_SPIRONOLACTONE', 'MED_CURRENT_MEMORY_AGENT',
    'Y00_HGB_A1C', 'Y00_TRIGLYCERIDE', 'Y00_HDL', 'Y00_LDL', 'Y00_CHOLESTEROL', 'Y00_HSCRP',
    'LAST_MED_ENC_TYPE'
]

TARGET_COLUMN = 'MACE'
CLASS_LABELS = ['No MACE', 'MACE']


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


class SklearnRAIWrapper:
    """
    A pure sklearn-compatible wrapper that stores prediction functions.
    
    This wrapper avoids pickling the AutoML model directly by only storing
    the prediction arrays (for a pre-computed approach) or by extracting
    the underlying sklearn estimator if available.
    
    For RAI compatibility, it provides:
    - predict(): Returns class labels (0 or 1)
    - predict_proba(): Returns probability estimates
    - classes_: Array of class labels
    - feature_names_in_: Feature names
    - n_features_in_: Number of features
    """
    
    def __init__(self, underlying_model, feature_names=None):
        """
        Initialize the wrapper by extracting sklearn-compatible components.
        
        Args:
            underlying_model: The AutoML model (or its underlying estimator)
            feature_names: List of feature names
        """
        self.feature_names = feature_names or FEATURE_NAMES
        self.feature_names_in_ = np.array(self.feature_names)
        self.n_features_in_ = len(self.feature_names)
        self.classes_ = np.array([0, 1])
        
        # Try to extract the underlying sklearn model
        self._model = self._extract_sklearn_model(underlying_model)
        self._has_predict_proba = hasattr(self._model, 'predict_proba')
        
        # Copy feature importances if available
        if hasattr(self._model, 'feature_importances_'):
            self.feature_importances_ = self._model.feature_importances_
    
    def _extract_sklearn_model(self, model):
        """
        Try to extract the underlying sklearn estimator from AutoML model.
        """
        # If it's already a sklearn model, return it
        if hasattr(model, 'predict') and hasattr(model, 'get_params'):
            # Check common AutoML wrapper attributes
            if hasattr(model, 'fitted_pipeline_'):
                return model.fitted_pipeline_
            if hasattr(model, 'model_'):
                return model.model_
            if hasattr(model, '_final_estimator'):
                return model._final_estimator
            if hasattr(model, 'steps'):
                # It's a pipeline, get the last step
                return model.steps[-1][1]
        return model
    
    def _prepare_input(self, X):
        """Convert input to appropriate format."""
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        else:
            return pd.DataFrame(X)
    
    def predict(self, X):
        """Predict class labels."""
        X = self._prepare_input(X)
        predictions = self._model.predict(X)
        # Ensure integer output
        if hasattr(predictions, 'astype'):
            return predictions.astype(int)
        return np.array(predictions).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = self._prepare_input(X)
        
        if self._has_predict_proba:
            proba = self._model.predict_proba(X)
            # Ensure 2D output
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            elif proba.shape[1] == 1:
                proba = np.column_stack([1 - proba.ravel(), proba.ravel()])
            return proba
        else:
            # Fallback: create pseudo-probabilities
            predictions = self.predict(X)
            proba = np.column_stack([
                np.where(predictions == 0, 0.95, 0.05),
                np.where(predictions == 1, 0.95, 0.05)
            ])
            return proba
    
    def get_params(self, deep=True):
        """Get parameters (sklearn compatibility)."""
        return {'feature_names': self.feature_names}
    
    def set_params(self, **params):
        """Set parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def load_automl_model(model_path: str, logger: logging.Logger):
    """
    Load the AutoML model from pickle file.
    
    Note: This may fail if the AutoML SDK version doesn't match.
    In that case, we'll try alternative approaches.
    """
    pkl_path = os.path.join(model_path, "model.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"model.pkl not found at {pkl_path}")
    
    logger.info(f"Loading model from {pkl_path}")
    
    try:
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        return model
    except ModuleNotFoundError as e:
        logger.warning(f"Failed to load AutoML model due to missing module: {e}")
        logger.warning("This is expected - AutoML models have internal dependencies")
        raise


def create_wrapped_model(model, output_path: str, logger: logging.Logger):
    """
    Create a wrapped model and save it in MLflow format.
    """
    logger.info("Creating sklearn-compatible wrapper...")
    
    wrapper = SklearnRAIWrapper(model, FEATURE_NAMES)
    
    logger.info(f"Wrapper created:")
    logger.info(f"  - n_features_in_: {wrapper.n_features_in_}")
    logger.info(f"  - classes_: {wrapper.classes_}")
    logger.info(f"  - has predict_proba: {wrapper._has_predict_proba}")
    
    # Create MLflow model
    logger.info(f"Saving MLflow model to {output_path}")
    
    # Define conda environment
    conda_env = {
        "name": "rai_model_env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.9",
            "pip",
            {
                "pip": [
                    "mlflow>=2.0",
                    "scikit-learn>=1.0",
                    "pandas>=1.5",
                    "numpy>=1.23",
                ]
            }
        ]
    }
    
    # Save the model
    mlflow.sklearn.save_model(
        wrapper,
        output_path,
        conda_env=conda_env,
    )
    
    logger.info("MLflow model saved successfully")
    return output_path


def register_model(ml_client: MLClient, model_path: str, model_name: str, logger: logging.Logger) -> Model:
    """
    Register the wrapped model in Azure ML.
    """
    logger.info(f"Registering model '{model_name}' from {model_path}")
    
    model = Model(
        name=model_name,
        path=model_path,
        type=AssetTypes.MLFLOW_MODEL,
        description="Secondary CVD Risk model wrapped for RAI dashboard compatibility",
    )
    
    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Model registered: {model_name}:{registered_model.version}")
    
    return registered_model


def get_rai_components(ml_client_registry: MLClient, logger: logging.Logger) -> dict:
    """Get RAI dashboard components from azureml registry."""
    logger.info("Fetching RAI components from azureml registry...")
    
    label = "latest"
    
    rai_constructor = ml_client_registry.components.get(
        name="rai_tabular_insight_constructor", label=label
    )
    version = rai_constructor.version
    logger.info(f"RAI components version: {version}")
    
    rai_explanation = ml_client_registry.components.get(
        name="rai_tabular_explanation", version=version
    )
    
    rai_gather = ml_client_registry.components.get(
        name="rai_tabular_insight_gather", version=version
    )
    
    return {
        'constructor': rai_constructor,
        'explanation': rai_explanation,
        'gather': rai_gather,
        'version': version
    }


def create_rai_explanation_pipeline(
    rai_components: dict,
    model: Model,
    train_data,
    test_data,
    compute_name: str,
    logger: logging.Logger
):
    """
    Create RAI explanation pipeline.
    """
    model_id = f"{model.name}:{model.version}"
    azureml_model_id = f"azureml:{model_id}"
    
    train_data_id = f"azureml:{train_data.name}:{train_data.version}"
    test_data_id = f"azureml:{test_data.name}:{test_data.version}"
    
    logger.info("Creating RAI explanation pipeline:")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Train data: {train_data_id}")
    logger.info(f"  Test data: {test_data_id}")
    
    rai_constructor = rai_components['constructor']
    rai_explanation = rai_components['explanation']
    rai_gather = rai_components['gather']
    
    @dsl.pipeline(
        compute=compute_name,
        description="RAI Explanation for Secondary CVD Risk model",
        experiment_name=f"RAI_explanation_{model.name}",
    )
    def rai_explanation_pipeline(target_column_name, train_data, test_data):
        # Step 1: Initialize RAI Insights
        create_rai_job = rai_constructor(
            title="RAI Explanation - Secondary CVD Risk",
            task_type="classification",
            model_info=model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
        )
        create_rai_job.set_limits(timeout=3600)
        
        # Step 2: Add Explanations
        explanation_job = rai_explanation(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            comment="SHAP feature explanations for Secondary CVD Risk model",
        )
        explanation_job.set_limits(timeout=3600)
        
        # Step 3: Gather insights
        rai_gather_job = rai_gather(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_4=explanation_job.outputs.explanation,
        )
        rai_gather_job.set_limits(timeout=3600)
        rai_gather_job.outputs.dashboard.mode = "upload"
        
        return {"dashboard": rai_gather_job.outputs.dashboard}
    
    # Create pipeline inputs
    import uuid
    
    train_input = Input(type="mltable", path=train_data_id, mode="download")
    test_input = Input(type="mltable", path=test_data_id, mode="download")
    
    pipeline_job = rai_explanation_pipeline(
        target_column_name=TARGET_COLUMN,
        train_data=train_input,
        test_data=test_input,
    )
    
    # Set output path
    rand_path = str(uuid.uuid4())
    pipeline_job.outputs.dashboard = Output(
        path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
        mode="upload",
        type="uri_folder",
    )
    
    return pipeline_job


def main():
    parser = argparse.ArgumentParser(description="Register wrapped model and run RAI explanation job")
    parser.add_argument("--model-path", default="./models/secondary_cvd_risk/1", 
                        help="Path to the model directory")
    parser.add_argument("--model-name", default="secondary-cvd-risk-rai-wrapped",
                        help="Name for the registered model")
    parser.add_argument("--compute", default="aml-cluster",
                        help="Compute cluster for RAI job")
    parser.add_argument("--data-name", default="secondary-cvd-risk",
                        help="Name of the data asset")
    parser.add_argument("--data-version", default="1",
                        help="Version of the data asset")
    parser.add_argument("--skip-register", action="store_true",
                        help="Skip model registration (use existing)")
    parser.add_argument("--skip-rai", action="store_true",
                        help="Skip RAI job submission")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 70)
    logger.info("Register RAI-Compatible Model and Run Explanation Job")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Get workspace details
    subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = os.getenv('AZURE_ML_WORKSPACE')
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Missing AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, or AZURE_ML_WORKSPACE")
    
    # Connect to Azure ML
    credential = AzureCliCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    logger.info(f"Connected to workspace: {workspace_name}")
    
    # Registry client for RAI components
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        registry_name="azureml",
    )
    
    registered_model = None
    
    if not args.skip_register:
        # Load and wrap the model
        model_path = os.path.abspath(args.model_path)
        
        try:
            automl_model = load_automl_model(model_path, logger)
            
            # Create wrapped model in temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                wrapped_model_path = os.path.join(tmpdir, "wrapped_model")
                create_wrapped_model(automl_model, wrapped_model_path, logger)
                
                # Register the model
                registered_model = register_model(ml_client, wrapped_model_path, args.model_name, logger)
                
        except ModuleNotFoundError as e:
            logger.error(f"Cannot load AutoML model: {e}")
            logger.error("")
            logger.error("=" * 70)
            logger.error("AUTOML MODEL LOADING FAILED")
            logger.error("=" * 70)
            logger.error("")
            logger.error("The AutoML model cannot be loaded because it references")
            logger.error("internal Azure ML SDK modules that aren't installed.")
            logger.error("")
            logger.error("Options:")
            logger.error("1. Run a NEW AutoML job with serverless compute and")
            logger.error("   enable_model_explainability=True to get built-in RAI")
            logger.error("")
            logger.error("2. Train a pure sklearn model instead of using AutoML")
            logger.error("")
            logger.error("See AUTOML_RAI_LIMITATIONS.md for details.")
            logger.error("=" * 70)
            return 1
    else:
        # Get existing model
        logger.info(f"Using existing model: {args.model_name}")
        try:
            models = list(ml_client.models.list(name=args.model_name))
            if models:
                registered_model = models[0]
                logger.info(f"Found model: {args.model_name}:{registered_model.version}")
            else:
                raise ValueError(f"Model '{args.model_name}' not found")
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return 1
    
    if args.skip_rai:
        logger.info("Skipping RAI job submission (--skip-rai)")
        return 0
    
    # Get data assets
    logger.info(f"Getting data assets: {args.data_name}:{args.data_version}")
    train_data = ml_client.data.get(name=args.data_name, version=args.data_version)
    test_data = ml_client.data.get(name=args.data_name, version=args.data_version)
    
    # Get RAI components
    rai_components = get_rai_components(ml_client_registry, logger)
    
    # Create and submit RAI pipeline
    pipeline_job = create_rai_explanation_pipeline(
        rai_components=rai_components,
        model=registered_model,
        train_data=train_data,
        test_data=test_data,
        compute_name=args.compute,
        logger=logger
    )
    
    logger.info("Submitting RAI explanation job...")
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("RAI Job Submitted")
    logger.info("=" * 70)
    logger.info(f"Job Name: {created_job.name}")
    logger.info(f"Status: {created_job.status}")
    logger.info(f"Studio URL: {created_job.studio_url}")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
