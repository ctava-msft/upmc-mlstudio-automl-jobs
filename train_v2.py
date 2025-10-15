"""
Azure Machine Learning AutoML Training Script (SDK v2)
Modern SDK with better storage integration and simplified APIs
"""

from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import automl
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, AzureCliCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import Data
import pandas as pd
import logging
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from feature_engineering import FeatureEngineer, validate_data

# Load environment variables from .env file
load_dotenv()


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration"""
    log_level = config.get('output', {}).get('log_level', 'INFO')
    log_to_file = config.get('output', {}).get('log_to_file', False)
    log_file_path = config.get('output', {}).get('log_file_path', './logs/training.log')
    
    # Create logs directory if needed
    if log_to_file:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file_path))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
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
    else:  # default
        logger.info("Using Default Azure Credential (tries multiple methods)")
        return DefaultAzureCredential()


def load_and_preprocess_data(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Load and preprocess data according to configuration"""
    data_config = config['data']
    input_path = data_config['input_path']
    
    logger.info(f"Loading data from {input_path}")
    
    # Determine file type and load appropriately
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    logger.info(f"Data loaded. Shape: {df.shape}")
    
    # Validate data
    validation_config = data_config.get('validation', {})
    if validation_config:
        logger.info("Validating data...")
        df = validate_data(df, validation_config)
        logger.info(f"Data validated. Shape after validation: {df.shape}")
    
    # Drop specified columns
    columns_to_drop = data_config.get('columns_to_drop', [])
    if columns_to_drop:
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        if existing_cols:
            df = df.drop(columns=existing_cols)
            logger.info(f"Dropped {len(existing_cols)} columns")
    
    # Apply feature engineering
    fe_config = config.get('feature_engineering', {})
    if fe_config.get('enabled', True):
        logger.info("Applying feature engineering...")
        engineer = FeatureEngineer(fe_config)
        df = engineer.apply_transformations(df)
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    
    return df


def get_or_create_data_asset(
    ml_client: MLClient,
    data_path: str,
    config: dict,
    logger: logging.Logger
) -> Data:
    """
    Get existing data asset or create new one
    SDK v2 handles storage authentication automatically!
    """
    data_config = config['data']
    dataset_name = data_config['dataset_name']
    use_existing = data_config.get('use_existing_dataset', False)
    
    if use_existing:
        # Get existing data asset by name
        dataset_version = data_config.get('dataset_version', 'latest')
        
        if dataset_version == 'latest':
            version_str = None  # SDK v2 will get latest
        else:
            version_str = str(dataset_version)
        
        logger.info(f"Getting existing data asset: {dataset_name} (version: {dataset_version})")
        
        try:
            if version_str:
                data_asset = ml_client.data.get(name=dataset_name, version=version_str)
            else:
                data_asset = ml_client.data.get(name=dataset_name, label="latest")
            
            logger.info(f"✓ Data asset obtained: {dataset_name} v{data_asset.version}")
            return data_asset
            
        except Exception as e:
            logger.error(f"Failed to get data asset '{dataset_name}': {str(e)}")
            logger.info("Tip: Verify the data asset exists in Azure ML Studio")
            raise
    
    # Create new data asset from local file
    logger.info(f"Creating new data asset from: {data_path}")
    
    data_asset = Data(
        name=dataset_name,
        description=data_config.get('dataset_description', 'Training dataset'),
        path=data_path,
        type=AssetTypes.URI_FILE
    )
    
    # SDK v2 automatically uploads and registers the data asset
    logger.info("Uploading and registering data asset...")
    data_asset = ml_client.data.create_or_update(data_asset)
    
    logger.info(f"✓ Data asset created: {dataset_name} v{data_asset.version}")
    return data_asset


def get_or_create_environment(
    ml_client: MLClient,
    env_name: str,
    conda_file_path: str,
    logger: logging.Logger
) -> Environment:
    """
    Get existing environment or create new one from conda file
    SDK v2 handles environment registration properly
    """
    env_version = "1"
    
    # Try to get existing environment
    try:
        logger.info(f"Checking for existing environment: {env_name}:{env_version}")
        env = ml_client.environments.get(name=env_name, version=env_version)
        logger.info(f"✓ Environment found: {env_name}:{env_version}")
        return env
    except Exception as e:
        logger.info(f"Environment not found, creating new one: {str(e)}")
    
    # Create new environment from conda file
    logger.info(f"Creating new environment from: {conda_file_path}")
    
    # Verify conda file exists
    if not os.path.exists(conda_file_path):
        raise FileNotFoundError(f"Conda file not found: {conda_file_path}")
    
    # Create environment with curated base image
    env = Environment(
        name=env_name,
        description="AutoML training environment with SDK v2",
        conda_file=conda_file_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",  # Curated base image
        version=env_version
    )
    
    logger.info("Registering environment with Azure ML workspace...")
    env = ml_client.environments.create_or_update(env)
    
    logger.info(f"✓ Environment created: {env_name}:{env.version}")
    return env


def create_automl_job(
    ml_client: MLClient,
    data_asset: Data,
    config: dict,
    logger: logging.Logger,
    environment: Environment = None
):
    """
    Create AutoML classification/regression/forecasting job using SDK v2
    """
    automl_cfg = config['automl']
    data_config = config['data']
    compute_config = config['compute']
    experiment_config = config['experiment']
    
    task_type = automl_cfg['task'].lower()
    
    logger.info(f"Creating AutoML {task_type} job")
    if environment:
        logger.info(f"Using environment: {environment.name}:{environment.version}")
    
    # Prepare training data input
    training_data_input = Input(
        type=AssetTypes.MLTABLE if data_asset.type == AssetTypes.MLTABLE else AssetTypes.URI_FILE,
        path=f"azureml:{data_asset.name}:{data_asset.version}"
    )
    
    # Create base job configuration based on task type
    if task_type == "classification":
        job = automl.classification(
            compute=compute_config['cluster_name'],
            experiment_name=experiment_config['name'],
            training_data=training_data_input,
            target_column_name=data_config['label_column'],
            primary_metric=automl_cfg['primary_metric'],
            n_cross_validations=automl_cfg['training'].get('n_cross_validations', 5),
            enable_model_explainability=True,
            tags={"framework": "AutoML", "sdk_version": "v2"}
        )
    elif task_type == "regression":
        job = automl.regression(
            compute=compute_config['cluster_name'],
            experiment_name=experiment_config['name'],
            training_data=training_data_input,
            target_column_name=data_config['label_column'],
            primary_metric=automl_cfg['primary_metric'],
            n_cross_validations=automl_cfg['training'].get('n_cross_validations', 5),
            enable_model_explainability=True,
            tags={"framework": "AutoML", "sdk_version": "v2"}
        )
    elif task_type == "forecasting":
        job = automl.forecasting(
            compute=compute_config['cluster_name'],
            experiment_name=experiment_config['name'],
            training_data=training_data_input,
            target_column_name=data_config['label_column'],
            primary_metric=automl_cfg['primary_metric'],
            n_cross_validations=automl_cfg['training'].get('n_cross_validations', 5),
            tags={"framework": "AutoML", "sdk_version": "v2"}
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Set training limits
    training_cfg = automl_cfg['training']
    job.set_limits(
        timeout_minutes=training_cfg.get('experiment_timeout_minutes', 60),
        trial_timeout_minutes=training_cfg.get('iteration_timeout_minutes', 20),
        max_trials=training_cfg.get('max_concurrent_iterations', 4),
        enable_early_termination=training_cfg.get('enable_early_stopping', True)
    )
    
    # Configure featurization
    featurization_cfg = automl_cfg.get('featurization', {})
    if featurization_cfg.get('mode') == 'off':
        job.set_featurization(enable_dnn_featurization=False)
    
    # Set allowed/blocked models
    models_cfg = automl_cfg.get('models', {})
    if models_cfg.get('allowed'):
        job.set_training(allowed_training_algorithms=models_cfg['allowed'])
    if models_cfg.get('blocked'):
        job.set_training(blocked_training_algorithms=models_cfg['blocked'])
    
    logger.info(f"AutoML job configured: {task_type.capitalize()}, Primary Metric: {automl_cfg['primary_metric']}")
    return job


def submit_job(ml_client: MLClient, job, logger: logging.Logger):
    """Submit AutoML job to Azure ML"""
    logger.info("Submitting AutoML job...")
    
    returned_job = ml_client.jobs.create_or_update(job)
    
    logger.info(f"✓ Job submitted successfully!")
    logger.info(f"  Job name: {returned_job.name}")
    logger.info(f"  Job ID: {returned_job.id}")
    logger.info(f"  Status: {returned_job.status}")
    logger.info(f"  Studio URL: {returned_job.studio_url}")
    
    return returned_job


def monitor_job(ml_client: MLClient, job, logger: logging.Logger):
    """Monitor job progress"""
    logger.info("=" * 70)
    logger.info("Job Monitoring")
    logger.info("=" * 70)
    logger.info("You can monitor the job in Azure ML Studio:")
    logger.info(f"  {job.studio_url}")
    logger.info("")
    logger.info("Waiting for job completion...")
    logger.info("(You can safely close this window and monitor in the portal)")
    logger.info("=" * 70)
    
    # Stream job logs (this will block until completion)
    ml_client.jobs.stream(job.name)
    
    # Get final job status
    job_status = ml_client.jobs.get(job.name)
    
    logger.info("=" * 70)
    logger.info(f"Job Status: {job_status.status}")
    logger.info("=" * 70)
    
    return job_status


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Azure ML AutoML Training Script (SDK v2)')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 70)
    logger.info("Azure ML AutoML Training (SDK v2)")
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info("=" * 70)
    
    try:
        # Get Azure credentials
        auth_config = config.get('authentication', {})
        auth_method = auth_config.get('method', 'default').lower()
        credential = get_credential(auth_method, logger)
        
        # Connect to Azure ML workspace
        logger.info("Connecting to Azure ML Workspace...")
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_ML_WORKSPACE")
        
        if not all([subscription_id, resource_group, workspace_name]):
            raise ValueError("Missing Azure credentials in .env file")
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        logger.info(f"✓ Connected to workspace: {workspace_name}")
        
        # Process data
        data_config = config['data']
        use_existing = data_config.get('use_existing_dataset', False)
        
        if use_existing:
            # Use existing data asset
            data_asset = get_or_create_data_asset(ml_client, None, config, logger)
        else:
            # Load and preprocess data from local file
            df = load_and_preprocess_data(config, logger)
            
            # Save preprocessed data if configured
            output_config = config.get('output', {})
            if output_config.get('save_preprocessed_data', True):
                preprocessed_dir = Path(output_config.get('preprocessed_data_path', './data/preprocessed'))
                preprocessed_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                preprocessed_path = preprocessed_dir / f"preprocessed_data_{timestamp}.csv"
                df.to_csv(preprocessed_path, index=False)
                logger.info(f"✓ Preprocessed data saved to: {preprocessed_path}")
            else:
                # Create temporary file
                preprocessed_path = f"temp_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(preprocessed_path, index=False)
            
            # Upload and register data asset (SDK v2 handles storage automatically!)
            data_asset = get_or_create_data_asset(ml_client, str(preprocessed_path), config, logger)
            
            # Clean up temp file if needed
            if not output_config.get('save_preprocessed_data', True):
                if Path(preprocessed_path).exists():
                    Path(preprocessed_path).unlink()
                    logger.info("Temporary file removed")
        
        # Get or create environment
        env_name = config.get('environment', {}).get('name', 'secondary-cvd-risk')
        conda_file = config.get('environment', {}).get('conda_file', 'conda_env_v_1_0_0.yml')
        environment = get_or_create_environment(ml_client, env_name, conda_file, logger)
        
        # Create AutoML job
        job = create_automl_job(ml_client, data_asset, config, logger, environment)
        
        # Submit job
        submitted_job = submit_job(ml_client, job, logger)
        
        # Monitor job (optional - can be skipped for async submission)
        monitor_config = config.get('output', {}).get('monitor_job', True)
        if monitor_config:
            final_job = monitor_job(ml_client, submitted_job, logger)
            
            if final_job.status == "Completed":
                logger.info("=" * 70)
                logger.info("✓ Training completed successfully!")
                logger.info("=" * 70)
                logger.info(f"Best model: {final_job.name}")
                logger.info(f"View results: {final_job.studio_url}")
            else:
                logger.warning(f"Job finished with status: {final_job.status}")
        else:
            logger.info("Job submitted. Not waiting for completion.")
            logger.info(f"Monitor at: {submitted_job.studio_url}")
        
        logger.info("=" * 70)
        logger.info("Training pipeline completed!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
