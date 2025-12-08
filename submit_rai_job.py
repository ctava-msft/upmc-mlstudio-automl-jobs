"""
Azure ML Responsible AI Dashboard Job Submission Script

This script creates and submits a RAI dashboard pipeline job for the 
Secondary CVD Risk model. It follows the pattern from:
https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/10/Create%20Responsible%20AI%20dashboard.ipynb

Components used:
- rai_tabular_insight_constructor: Initializes RAI dashboard
- rai_tabular_erroranalysis: Error analysis component
- rai_tabular_explanation: Model explanations (SHAP)
- rai_tabular_insight_gather: Gathers all insights for visualization
"""

import os
import sys
import uuid
import time
import yaml
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Azure ML imports
from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.entities import Model, PipelineJob
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, AzureCliCredential, InteractiveBrowserCredential

# Load environment variables
load_dotenv()


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "rai_config.yaml") -> dict:
    """Load RAI configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_credential(auth_method: str, logger: logging.Logger):
    """Get appropriate Azure credential based on configuration"""
    logger.info(f"Authentication method: {auth_method}")
    
    if auth_method == "cli":
        logger.info("Using Azure CLI authentication")
        return AzureCliCredential()
    elif auth_method == "interactive":
        logger.info("Using Interactive Browser authentication")
        return InteractiveBrowserCredential()
    else:
        logger.info("Using Default Azure Credential")
        return DefaultAzureCredential()


def get_ml_client(config: dict, logger: logging.Logger) -> tuple:
    """Create MLClient for Azure ML workspace and return with credential"""
    auth_method = config.get('authentication', {}).get('method', 'cli')
    credential = get_credential(auth_method, logger)
    
    # Get workspace details from environment or config
    subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID') or config.get('workspace', {}).get('subscription_id')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP') or config.get('workspace', {}).get('resource_group')
    workspace_name = os.getenv('AZURE_ML_WORKSPACE') or config.get('workspace', {}).get('name')
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Missing workspace configuration. Set environment variables or config:\n"
            "  AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE"
        )
    
    logger.info(f"Connecting to workspace: {workspace_name}")
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    return ml_client, credential


def get_registry_client(ml_client: MLClient, credential, logger: logging.Logger) -> MLClient:
    """Get MLClient for azureml registry to access RAI components"""
    logger.info("Connecting to azureml registry for RAI components...")
    
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=ml_client.subscription_id,
        resource_group_name=ml_client.resource_group_name,
        registry_name="azureml",
    )
    
    return ml_client_registry


def register_model(ml_client: MLClient, config: dict, logger: logging.Logger) -> Model:
    """Register or get the model for RAI dashboard"""
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'secondary-cvd-risk-rai')
    model_path = model_config.get('path', './models/secondary_cvd_risk/1')
    force_reregister = model_config.get('force_reregister', False)
    
    # Convert to absolute path
    model_path = os.path.abspath(model_path)
    
    if force_reregister:
        logger.info(f"Force re-registering model: {model_name}")
        logger.info(f"Model path: {model_path}")
        
        # Verify MLmodel file exists
        mlmodel_path = os.path.join(model_path, "MLmodel")
        if not os.path.exists(mlmodel_path):
            raise ValueError(f"MLmodel file not found at {mlmodel_path}")
        logger.info(f"MLmodel file found: {mlmodel_path}")
        
        # Register as MLflow model
        model = Model(
            name=model_name,
            path=model_path,
            type=AssetTypes.MLFLOW_MODEL,  # MLflow format for RAI
            description="Secondary CVD Risk AutoML model in MLflow format for RAI dashboard",
        )
        registered_model = ml_client.models.create_or_update(model)
        logger.info(f"Model registered: {model_name}:{registered_model.version}")
        return registered_model
    
    # Check if model already exists
    try:
        model_version = model_config.get('version', None)
        if model_version:
            model = ml_client.models.get(name=model_name, version=model_version)
            logger.info(f"Using existing model: {model_name}:{model.version}")
            return model
        else:
            # Get latest version
            models = list(ml_client.models.list(name=model_name))
            if models:
                model = models[0]
                logger.info(f"Using existing model: {model_name}:{model.version}")
                return model
    except Exception as e:
        logger.error(f"Model not found: {model_name}")
        raise ValueError(f"Model '{model_name}' must be registered before running RAI job. Error: {e}")
    
    raise ValueError(f"Model '{model_name}' not found. Please register the model first.")


def get_data_assets(ml_client: MLClient, config: dict, logger: logging.Logger):
    """Get existing train/test data assets for RAI dashboard"""
    data_config = config.get('data', {})
    
    train_data_name = data_config.get('train_data_name', 'secondary-cvd-risk')
    test_data_name = data_config.get('test_data_name', 'secondary-cvd-risk')
    data_version = data_config.get('version', '1')
    
    # Get existing data assets
    logger.info(f"Getting existing train data: {train_data_name}:{data_version}")
    train_data = ml_client.data.get(name=train_data_name, version=data_version)
    logger.info(f"Train data found: {train_data.name}:{train_data.version}")
    
    logger.info(f"Getting existing test data: {test_data_name}:{data_version}")
    test_data = ml_client.data.get(name=test_data_name, version=data_version)
    logger.info(f"Test data found: {test_data.name}:{test_data.version}")
    
    return train_data, test_data


def get_rai_components(ml_client_registry: MLClient, logger: logging.Logger):
    """Get RAI dashboard components from azureml registry"""
    logger.info("Fetching RAI components from azureml registry...")
    
    label = "latest"
    
    # Get constructor component and its version
    rai_constructor_component = ml_client_registry.components.get(
        name="rai_tabular_insight_constructor", label=label
    )
    version = rai_constructor_component.version
    logger.info(f"RAI components version: {version}")
    
    # Get other components with same version
    rai_erroranalysis_component = ml_client_registry.components.get(
        name="rai_tabular_erroranalysis", version=version
    )
    
    rai_explanation_component = ml_client_registry.components.get(
        name="rai_tabular_explanation", version=version
    )
    
    rai_gather_component = ml_client_registry.components.get(
        name="rai_tabular_insight_gather", version=version
    )
    
    logger.info("RAI components retrieved successfully")
    
    return {
        'constructor': rai_constructor_component,
        'erroranalysis': rai_erroranalysis_component,
        'explanation': rai_explanation_component,
        'gather': rai_gather_component,
        'version': version
    }


def create_rai_pipeline(
    rai_components: dict,
    model: Model,
    train_data,
    test_data,
    config: dict,
    logger: logging.Logger
):
    """Create RAI dashboard pipeline"""
    
    pipeline_config = config.get('pipeline', {})
    compute_name = pipeline_config.get('compute', 'aml-cluster')
    target_column = config.get('data', {}).get('target_column', 'MACE')
    
    model_name = model.name
    model_version = model.version
    expected_model_id = f"{model_name}:{model_version}"
    azureml_model_id = f"azureml:{expected_model_id}"
    
    train_data_id = f"azureml:{train_data.name}:{train_data.version}"
    test_data_id = f"azureml:{test_data.name}:{test_data.version}"
    
    logger.info("Creating RAI pipeline with:")
    logger.info("  Model: %s", expected_model_id)
    logger.info(f"  Train data: {train_data_id}")
    logger.info(f"  Test data: {test_data_id}")
    logger.info(f"  Target column: {target_column}")
    logger.info(f"  Compute: {compute_name}")
    
    rai_constructor = rai_components['constructor']
    rai_erroranalysis = rai_components['erroranalysis']
    rai_explanation = rai_components['explanation']
    rai_gather = rai_components['gather']
    
    @dsl.pipeline(
        compute=compute_name,
        description="RAI insights on Secondary CVD Risk data",
        experiment_name=f"RAI_insights_{model_name}",
    )
    def rai_decision_pipeline(
        target_column_name, train_data, test_data
    ):
        # Step 1: Initiate the RAI Insights
        create_rai_job = rai_constructor(
            title="RAI Dashboard - Secondary CVD Risk",
            task_type="classification",
            model_info=expected_model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
            use_model_dependency=True,  # Install model's conda dependencies
        )
        create_rai_job.set_limits(timeout=3600)  # 1 hour
        
        # Step 2: Add Error Analysis
        error_job = rai_erroranalysis(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        error_job.set_limits(timeout=3600)  # 1 hour
        
        # Step 3: Add Explanations
        explanation_job = rai_explanation(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            comment="SHAP feature explanations for Secondary CVD Risk model",
        )
        explanation_job.set_limits(timeout=3600)  # 1 hour
        
        # Step 4: Gather all insights
        rai_gather_job = rai_gather(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_3=error_job.outputs.error_analysis,
            insight_4=explanation_job.outputs.explanation,
        )
        rai_gather_job.set_limits(timeout=3600)  # 1 hour
        
        rai_gather_job.outputs.dashboard.mode = "upload"
        
        return {
            "dashboard": rai_gather_job.outputs.dashboard,
        }
    
    # Create pipeline inputs
    train_input = Input(
        type="mltable",
        path=train_data_id,
        mode="download",
    )
    test_input = Input(
        type="mltable",
        path=test_data_id,
        mode="download",
    )
    
    # Build the pipeline
    insights_pipeline_job = rai_decision_pipeline(
        target_column_name=target_column,
        train_data=train_input,
        test_data=test_input,
    )
    
    # Set output path
    rand_path = str(uuid.uuid4())
    insights_pipeline_job.outputs.dashboard = Output(
        path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
        mode="upload",
        type="uri_folder",
    )
    
    return insights_pipeline_job


def submit_and_wait(ml_client: MLClient, pipeline_job: PipelineJob, logger: logging.Logger, wait: bool = True) -> PipelineJob:
    """Submit pipeline job and optionally wait for completion"""
    logger.info("Submitting RAI pipeline job...")
    
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    
    logger.info(f"Pipeline job submitted: {created_job.name}")
    logger.info(f"Studio URL: {created_job.studio_url}")
    
    if not wait:
        logger.info("Job submitted. Use --wait to monitor progress.")
        return created_job
    
    logger.info("Waiting for job completion...")
    
    terminal_states = ["Completed", "Failed", "Canceled", "NotResponding"]
    
    while created_job.status not in terminal_states:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        logger.info(f"Job status: {created_job.status}")
    
    if created_job.status == "Completed":
        logger.info("✅ RAI pipeline completed successfully!")
        logger.info(f"View dashboard at: {created_job.studio_url}")
    else:
        logger.error(f"❌ RAI pipeline ended with status: {created_job.status}")
    
    return created_job


def main():
    parser = argparse.ArgumentParser(description="Submit RAI Dashboard job to Azure ML")
    parser.add_argument("--config", default="rai_config.yaml", help="Path to RAI config file")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 70)
    logger.info("Azure ML Responsible AI Dashboard Job Submission")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Get ML clients
    ml_client, credential = get_ml_client(config, logger)
    
    ml_client_registry = get_registry_client(ml_client, credential, logger)
    
    # Register or get model
    model = register_model(ml_client, config, logger)
    
    # Get existing data assets
    train_data, test_data = get_data_assets(ml_client, config, logger)
    
    # Get RAI components
    rai_components = get_rai_components(ml_client_registry, logger)
    
    # Create RAI pipeline
    pipeline_job = create_rai_pipeline(
        rai_components=rai_components,
        model=model,
        train_data=train_data,
        test_data=test_data,
        config=config,
        logger=logger
    )
    
    # Submit and optionally wait
    job = submit_and_wait(ml_client, pipeline_job, logger, wait=args.wait)
    
    # Output results
    logger.info("")
    logger.info("=" * 70)
    logger.info("RAI Job Submission Summary")
    logger.info("=" * 70)
    logger.info(f"Job Name: {job.name}")
    logger.info(f"Status: {job.status}")
    logger.info(f"Studio URL: {job.studio_url}")
    logger.info("=" * 70)
    
    return 0 if job.status in ["Completed", "Running", "Preparing", "Queued", "NotStarted", "Starting"] else 1


if __name__ == "__main__":
    sys.exit(main())
