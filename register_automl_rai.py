"""
Register AutoML Model with Wrapper and Run RAI Job

This script:
1. Creates a wrapped MLflow model that includes sklearn-compatible wrapper code
2. Registers the model in Azure ML
3. Creates/uses a custom environment with AutoML dependencies
4. Submits RAI explanation job using the custom environment

The key insight is that the RAI components run on Azure ML compute where
the custom environment with AutoML SDK can properly load the model.
"""

import os
import sys
import shutil
import argparse
import logging
import uuid
import time
import tempfile
from datetime import datetime
from dotenv import load_dotenv

from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.entities import Model, Environment, BuildContext
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential

load_dotenv()

# Configuration
TARGET_COLUMN = 'MACE'
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


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def get_wrapper_code(feature_names_list: list) -> str:
    """Generate the wrapper code with feature names embedded."""
    feature_names_str = repr(feature_names_list)
    
    # Use string concatenation to avoid f-string escaping issues with curly braces
    code = '''"""
RAI-Compatible Sklearn Wrapper for AutoML Model

This module provides a sklearn-compatible wrapper that:
- Loads the AutoML model using the AutoML runtime
- Implements predict() and predict_proba() methods
- Provides feature_names_in_ and classes_ attributes for RAI components

IMPORTANT: This module implements _load_pyfunc() for MLflow python_function flavor.
"""

import os
import pickle
import numpy as np
import pandas as pd

# Feature names embedded at model packaging time
FEATURE_NAMES = ''' + feature_names_str + '''


class SklearnAutoMLWrapper:
    """
    Sklearn-compatible wrapper for AutoML models.
    
    This wrapper loads the underlying AutoML model and provides
    the interface required by RAI dashboard components.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the wrapper.
        
        Args:
            model_path: Path to the model.pkl file. If None, will look for
                       model.pkl in the same directory as this module.
        """
        if model_path is None:
            # Look for model.pkl in the same directory
            module_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(module_dir, "model.pkl")
        
        # Load the AutoML model
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        
        # Set sklearn-compatible attributes
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.array(FEATURE_NAMES)
        self.n_features_in_ = len(FEATURE_NAMES)
        
        # Check capabilities
        self._has_predict_proba = hasattr(self._model, 'predict_proba')
        
        # Copy feature importances if available
        if hasattr(self._model, 'feature_importances_'):
            self.feature_importances_ = self._model.feature_importances_
    
    def _prepare_input(self, X):
        """Convert input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            # Use available feature names for the columns we have
            n_cols = X.shape[1]
            col_names = FEATURE_NAMES[:n_cols] if n_cols <= len(FEATURE_NAMES) else ["f" + str(i) for i in range(n_cols)]
            return pd.DataFrame(X, columns=col_names)
        else:
            return pd.DataFrame(X)
    
    def predict(self, X):
        """Predict class labels."""
        X = self._prepare_input(X)
        predictions = self._model.predict(X)
        # Ensure integer output
        if hasattr(predictions, 'dtype') and predictions.dtype == bool:
            return predictions.astype(int)
        return np.array(predictions).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = self._prepare_input(X)
        
        if self._has_predict_proba:
            proba = self._model.predict_proba(X)
            # Ensure 2D output for binary classification
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            elif proba.shape[1] == 1:
                proba = np.column_stack([1 - proba.ravel(), proba.ravel()])
            return proba
        else:
            # Fallback: create pseudo-probabilities from predictions
            predictions = self.predict(X)
            proba = np.column_stack([
                np.where(predictions == 0, 0.95, 0.05),
                np.where(predictions == 1, 0.95, 0.05)
            ])
            return proba
    
    def get_params(self, deep=True):
        """Get parameters (sklearn compatibility)."""
        return {}
    
    def set_params(self, **params):
        """Set parameters (sklearn compatibility)."""
        return self


def _load_pyfunc(data_path):
    """
    MLflow python_function loader function.
    
    This function is called by MLflow when loading the model with the python_function flavor.
    The data_path parameter points to the model artifact directory.
    
    Args:
        data_path: Path to the model artifacts directory (where model.pkl is located)
    
    Returns:
        SklearnAutoMLWrapper instance that implements predict()
    """
    model_pkl_path = os.path.join(data_path, "model.pkl")
    return SklearnAutoMLWrapper(model_pkl_path)


def load_model(model_path=None):
    """
    Load the wrapped model.
    
    Args:
        model_path: Path to model.pkl. If None, looks in module directory.
    
    Returns:
        SklearnAutoMLWrapper instance
    """
    return SklearnAutoMLWrapper(model_path)
'''
    return code


def create_wrapped_model_package(source_model_path: str, output_path: str, logger: logging.Logger):
    """
    Create a wrapped model package that includes:
    - The original model.pkl
    - A wrapper module for sklearn compatibility
    - MLmodel file pointing to the wrapper
    - conda.yaml with AutoML dependencies
    """
    logger.info(f"Creating wrapped model package from {source_model_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy model.pkl
    src_pkl = os.path.join(source_model_path, "model.pkl")
    dst_pkl = os.path.join(output_path, "model.pkl")
    shutil.copy2(src_pkl, dst_pkl)
    logger.info(f"Copied model.pkl")
    
    # Create wrapper module
    wrapper_code = get_wrapper_code(FEATURE_NAMES)
    wrapper_path = os.path.join(output_path, "wrapper.py")
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    logger.info("Created wrapper.py")
    
    # Create code directory with __init__.py for proper module import
    code_dir = os.path.join(output_path, "code")
    os.makedirs(code_dir, exist_ok=True)
    
    # Copy wrapper to code directory as well
    wrapper_code_path = os.path.join(code_dir, "wrapper.py")
    with open(wrapper_code_path, 'w') as f:
        f.write(wrapper_code)
    
    # Create __init__.py
    init_path = os.path.join(code_dir, "__init__.py")
    with open(init_path, 'w') as f:
        f.write("# Auto-generated init file\n")
    logger.info("Created code directory with wrapper module")
    
    # Create MLmodel file - using python_function flavor with code path
    # The loader_module must have a _load_pyfunc(data_path) function
    mlmodel_content = '''artifact_path: .
flavors:
  python_function:
    code: code
    data: .
    env:
      conda: conda.yaml
    loader_module: wrapper
    python_version: "3.9"
mlflow_version: "2.9.2"
model_uuid: automl-rai-wrapped
'''
    mlmodel_path = os.path.join(output_path, "MLmodel")
    with open(mlmodel_path, 'w') as f:
        f.write(mlmodel_content)
    logger.info("Created MLmodel")
    
    # Create conda.yaml with AutoML dependencies
    conda_content = '''name: automl_rai_env
channels:
  - conda-forge
  - anaconda
dependencies:
  - python=3.9
  - pip=24.2
  - cmake>=3.27
  - libgomp
  - numpy=1.23.5
  - pandas=1.5.3
  - scikit-learn=1.5.1
  - pip:
    - azureml-train-automl-runtime==1.60.0
    - azureml-interpret==1.60.0
    - azureml-defaults==1.60.0
    - mlflow>=2.9.0
    - inference-schema
'''
    conda_path = os.path.join(output_path, "conda.yaml")
    with open(conda_path, 'w') as f:
        f.write(conda_content)
    logger.info("Created conda.yaml")
    
    logger.info(f"Wrapped model package created at {output_path}")
    return output_path


def create_or_get_environment(ml_client: MLClient, env_name: str, conda_file: str, logger: logging.Logger) -> Environment:
    """
    Create or get an existing custom environment with AutoML dependencies.
    """
    logger.info(f"Creating/getting environment: {env_name}")
    
    # Check if environment already exists
    try:
        existing_envs = list(ml_client.environments.list(name=env_name))
        if existing_envs:
            env = existing_envs[0]
            logger.info(f"Using existing environment: {env_name}:{env.version}")
            return env
    except Exception:
        pass
    
    # Create new environment from conda file
    env = Environment(
        name=env_name,
        description="AutoML environment with RAI support for Secondary CVD Risk model",
        conda_file=conda_file,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    
    registered_env = ml_client.environments.create_or_update(env)
    logger.info(f"Environment created: {env_name}:{registered_env.version}")
    
    return registered_env


def register_model(ml_client: MLClient, model_path: str, model_name: str, logger: logging.Logger) -> Model:
    """Register the wrapped model in Azure ML."""
    logger.info(f"Registering model '{model_name}' from {model_path}")
    
    model = Model(
        name=model_name,
        path=model_path,
        type=AssetTypes.MLFLOW_MODEL,
        description="Secondary CVD Risk AutoML model with sklearn wrapper for RAI compatibility",
    )
    
    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Model registered: {model_name}:{registered_model.version}")
    
    return registered_model


def get_rai_components(ml_client_registry: MLClient, logger: logging.Logger) -> dict:
    """Get RAI components from azureml registry."""
    logger.info("Fetching RAI components from azureml registry...")
    
    rai_constructor = ml_client_registry.components.get(
        name="rai_tabular_insight_constructor", label="latest"
    )
    version = rai_constructor.version
    logger.info(f"RAI components version: {version}")
    
    rai_erroranalysis = ml_client_registry.components.get(
        name="rai_tabular_erroranalysis", version=version
    )
    
    rai_explanation = ml_client_registry.components.get(
        name="rai_tabular_explanation", version=version
    )
    
    rai_gather = ml_client_registry.components.get(
        name="rai_tabular_insight_gather", version=version
    )
    
    return {
        'constructor': rai_constructor,
        'erroranalysis': rai_erroranalysis,
        'explanation': rai_explanation,
        'gather': rai_gather,
        'version': version
    }


def create_rai_pipeline(
    rai_components: dict,
    model: Model,
    train_data,
    test_data,
    compute_name: str,
    logger: logging.Logger
):
    """
    Create RAI dashboard pipeline with error analysis and explanations.
    
    The pipeline uses the model's embedded conda environment which has
    AutoML dependencies, allowing it to load the model properly.
    """
    model_id = f"{model.name}:{model.version}"
    azureml_model_id = f"azureml:{model_id}"
    train_data_id = f"azureml:{train_data.name}:{train_data.version}"
    test_data_id = f"azureml:{test_data.name}:{test_data.version}"
    
    logger.info("Creating RAI pipeline:")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Train data: {train_data_id}")
    logger.info(f"  Test data: {test_data_id}")
    logger.info(f"  Compute: {compute_name}")
    
    rai_constructor = rai_components['constructor']
    rai_erroranalysis = rai_components['erroranalysis']
    rai_explanation = rai_components['explanation']
    rai_gather = rai_components['gather']
    
    @dsl.pipeline(
        compute=compute_name,
        description="RAI Dashboard for Secondary CVD Risk AutoML model",
        experiment_name=f"RAI_AutoML_{model.name}",
    )
    def rai_automl_pipeline(target_column_name, train_data, test_data):
        # Step 1: Initialize RAI Insights
        # use_model_dependency=True tells RAI to use the model's conda.yaml
        create_rai_job = rai_constructor(
            title="RAI Dashboard - Secondary CVD Risk (AutoML)",
            task_type="classification",
            model_info=model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
            use_model_dependency=True,  # Use model's conda.yaml for dependencies
        )
        create_rai_job.set_limits(timeout=7200)  # 2 hours
        
        # Step 2: Error Analysis
        error_job = rai_erroranalysis(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        error_job.set_limits(timeout=3600)
        
        # Step 3: Model Explanations (SHAP)
        explanation_job = rai_explanation(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            comment="SHAP feature explanations for Secondary CVD Risk AutoML model",
        )
        explanation_job.set_limits(timeout=3600)
        
        # Step 4: Gather all insights
        rai_gather_job = rai_gather(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_3=error_job.outputs.error_analysis,
            insight_4=explanation_job.outputs.explanation,
        )
        rai_gather_job.set_limits(timeout=3600)
        rai_gather_job.outputs.dashboard.mode = "upload"
        
        return {"dashboard": rai_gather_job.outputs.dashboard}
    
    # Create pipeline inputs
    train_input = Input(type="mltable", path=train_data_id, mode="download")
    test_input = Input(type="mltable", path=test_data_id, mode="download")
    
    pipeline_job = rai_automl_pipeline(
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
    parser = argparse.ArgumentParser(
        description="Register AutoML model with wrapper and run RAI job"
    )
    parser.add_argument(
        "--model-path", 
        default="./models/secondary_cvd_risk/1",
        help="Path to the AutoML model directory"
    )
    parser.add_argument(
        "--model-name", 
        default="secondary-cvd-risk-rai",
        help="Name for the registered model"
    )
    parser.add_argument(
        "--env-name",
        default="automl-rai-env",
        help="Name for the custom environment"
    )
    parser.add_argument(
        "--compute", 
        default="aml-cluster",
        help="Compute cluster name"
    )
    parser.add_argument(
        "--data-name", 
        default="secondary-cvd-risk",
        help="Data asset name"
    )
    parser.add_argument(
        "--data-version", 
        default="1",
        help="Data asset version"
    )
    parser.add_argument(
        "--skip-register", 
        action="store_true",
        help="Skip model registration, use existing"
    )
    parser.add_argument(
        "--skip-rai", 
        action="store_true",
        help="Skip RAI job submission"
    )
    parser.add_argument(
        "--wait", 
        action="store_true",
        help="Wait for RAI job completion"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO"
    )
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 70)
    logger.info("Register AutoML Model with Wrapper and Run RAI Job")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Get workspace details
    subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = os.getenv('AZURE_ML_WORKSPACE')
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Missing Azure ML environment variables")
    
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
        # Create wrapped model package
        model_path = os.path.abspath(args.model_path)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapped_path = os.path.join(tmpdir, "wrapped_model")
            create_wrapped_model_package(model_path, wrapped_path, logger)
            
            # Register the wrapped model
            registered_model = register_model(ml_client, wrapped_path, args.model_name, logger)
    else:
        # Get existing model
        logger.info(f"Using existing model: {args.model_name}")
        try:
            models = list(ml_client.models.list(name=args.model_name))
            if models:
                registered_model = models[0]
                logger.info(f"Found: {args.model_name}:{registered_model.version}")
            else:
                raise ValueError(f"Model '{args.model_name}' not found")
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return 1
    
    if args.skip_rai:
        logger.info("Skipping RAI job (--skip-rai)")
        logger.info("")
        logger.info("=" * 70)
        logger.info("Model Registration Complete")
        logger.info("=" * 70)
        logger.info(f"Model: {registered_model.name}:{registered_model.version}")
        logger.info("=" * 70)
        return 0
    
    # Get data assets
    logger.info(f"Getting data assets: {args.data_name}:{args.data_version}")
    train_data = ml_client.data.get(name=args.data_name, version=args.data_version)
    test_data = ml_client.data.get(name=args.data_name, version=args.data_version)
    
    # Get RAI components
    rai_components = get_rai_components(ml_client_registry, logger)
    
    # Create RAI pipeline
    pipeline_job = create_rai_pipeline(
        rai_components=rai_components,
        model=registered_model,
        train_data=train_data,
        test_data=test_data,
        compute_name=args.compute,
        logger=logger
    )
    
    # Submit job
    logger.info("Submitting RAI pipeline job...")
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("RAI Pipeline Job Submitted")
    logger.info("=" * 70)
    logger.info(f"Job Name: {created_job.name}")
    logger.info(f"Status: {created_job.status}")
    logger.info(f"Studio URL: {created_job.studio_url}")
    logger.info("=" * 70)
    
    if args.wait:
        logger.info("Waiting for job completion...")
        terminal_states = ["Completed", "Failed", "Canceled", "NotResponding"]
        
        while created_job.status not in terminal_states:
            time.sleep(60)
            created_job = ml_client.jobs.get(created_job.name)
            logger.info(f"Status: {created_job.status}")
        
        if created_job.status == "Completed":
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ RAI Dashboard Job Completed!")
            logger.info("=" * 70)
            logger.info(f"View dashboard: {created_job.studio_url}")
            logger.info("=" * 70)
        else:
            logger.error(f"❌ Job ended with status: {created_job.status}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
