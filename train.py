"""
Azure Machine Learning AutoML Training Script
Generic, configurable training script for any dataset and model type
"""

from azureml.core import Workspace, Dataset, Experiment, Model
from azureml.core.authentication import MsiAuthentication, InteractiveLoginAuthentication
from azureml.train.automl import AutoMLConfig
try:
    from azureml.automl.core.featurization import FeaturizationConfig
except ImportError:
    # Fallback for older SDK versions
    from azureml.train.automl.featurization import FeaturizationConfig
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, InteractiveBrowserCredential
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
    """
    Load training configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_preprocess_data(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and preprocess data according to configuration
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Preprocessed DataFrame
    """
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


def get_or_create_dataset(
    ws: Workspace, 
    data_path: str, 
    config: dict,
    logger: logging.Logger
) -> Dataset:
    """
    Get existing dataset or upload and register new dataset
    
    Args:
        ws: Azure ML Workspace
        data_path: Path to local data file (used only if not using existing dataset)
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Azure ML Dataset (existing or newly registered)
    """
    data_config = config['data']
    dataset_name = data_config['dataset_name']
    dataset_description = data_config.get('dataset_description', 'Training dataset')
    use_existing_dataset = data_config.get('use_existing_dataset', False)
    
    # Option 1: Use existing registered dataset
    if use_existing_dataset:
        dataset_id = data_config.get('dataset_id')
        dataset_version = data_config.get('dataset_version')
        
        logger.info(f"Using existing dataset: {dataset_name}")
        logger.info("Note: Dataset will be accessed by Azure compute during training")
        
        # Try using dataset_id first (most direct, avoids metadata download)
        if dataset_id:
            logger.info(f"Using dataset ID: {dataset_id}")
            try:
                # Parse the ID format: azureml:name:version
                if dataset_id.startswith("azureml:"):
                    parts = dataset_id.split(":")
                    if len(parts) == 3:
                        _, ds_name, ds_version = parts
                        dataset = Dataset.get_by_name(ws, name=ds_name, version=int(ds_version))
                        logger.info(f"Dataset obtained via ID: {ds_name} v{ds_version}")
                        return dataset
            except Exception as e:
                logger.warning(f"Could not use dataset_id, trying name/version: {str(e)[:200]}")
        
        # Fallback to name/version
        try:
            if not dataset_version or dataset_version == 'latest':
                logger.info(f"Attempting to get latest version of: {dataset_name}")
                dataset = Dataset.get_by_name(ws, name=dataset_name)
            else:
                logger.info(f"Attempting to get {dataset_name} version: {dataset_version}")
                dataset = Dataset.get_by_name(ws, name=dataset_name, version=int(dataset_version))
            
            logger.info(f"Successfully obtained dataset reference")
            return dataset
            
        except Exception as e:
            # If metadata download fails due to storage auth, try creating a reference-only dataset
            # This will be resolved by Azure compute during training execution
            logger.warning(f"Failed to download dataset metadata locally: {str(e)[:200]}")
            logger.info("Creating dataset reference for Azure compute (metadata will be accessed during training)")
            
            try:
                # Try getting dataset ID without downloading data
                from azureml.core import Datastore
                from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
                
                # Create a consumption config that references the dataset by name
                # This avoids local metadata download and will be resolved on Azure compute
                version_str = f":{dataset_version}" if dataset_version and dataset_version != 'latest' else ""
                dataset_id = f"{dataset_name}{version_str}"
                
                logger.info(f"✓ Created reference to dataset: {dataset_id}")
                logger.info("  Dataset will be accessed by Azure compute cluster during training")
                logger.info("  (Note: Local validation skipped due to storage authentication)")
                
                # Return a dictionary with dataset reference info instead of Dataset object
                # The AutoML config will accept this and resolve it on Azure compute
                return {
                    'dataset_name': dataset_name,
                    'dataset_version': dataset_version,
                    'workspace': ws
                }
                
            except Exception as ref_error:
                logger.error(f"Could not create dataset reference: {str(ref_error)[:200]}")
                # Fall through to original error handling
                pass
            
        except Exception as e:
            error_msg = str(e)
            logger.error("=" * 70)
            logger.error("STORAGE AUTHENTICATION REQUIRED")
            logger.error("=" * 70)
            logger.error(f"Cannot load dataset '{dataset_name}' due to storage authentication.")
            logger.error("")
            logger.error("Azure ML SDK v1 requires storage credentials to access dataset metadata,")
            logger.error("even when just creating a training configuration.")
            logger.error("")
            logger.error("SOLUTIONS:")
            logger.error("1. Configure Azure CLI authentication:")
            logger.error("   az login")
            logger.error(f"   az account set --subscription {ws.subscription_id}")
            logger.error("")
            logger.error("2. Configure storage account access in Azure Portal")
            logger.error("   - Navigate to the storage account for this workspace")
            logger.error("   - Grant your account 'Storage Blob Data Contributor' role")
            logger.error("")
            logger.error("3. Use local file upload instead:")
            logger.error("   - Set use_existing_dataset: false in config.yaml")
            logger.error("   - Provide input_path to local CSV file")
            logger.error("=" * 70)
            
            raise RuntimeError(
                f"Cannot access dataset '{dataset_name}' - storage authentication required. "
                f"See error details above for solutions."
            ) from e
    
    # Option 2: Upload and register new dataset from local file
    upload_to_datastore = data_config.get('upload_to_datastore', False)
    
    file_extension = Path(data_path).suffix
    
    # Check if upload is enabled
    if upload_to_datastore:
        logger.info("Uploading data to datastore (upload_to_datastore=true)")
        datastore = ws.get_default_datastore()
        
        # Get filename from path
        filename = Path(data_path).name
        target_path = f'data/{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        
        try:
            datastore.upload_files([data_path], target_path=target_path, overwrite=True)
            logger.info("Data uploaded successfully to datastore")
            
            # Create dataset from datastore path
            if file_extension in ['.csv', '.tsv']:
                dataset = Dataset.Tabular.from_delimited_files(
                    path=[(datastore, target_path + filename)]
                )
            elif file_extension == '.parquet':
                dataset = Dataset.Tabular.from_parquet_files(
                    path=[(datastore, target_path + filename)]
                )
            else:
                raise ValueError(f"Unsupported file format for dataset registration: {file_extension}")
                
        except Exception as e:
            logger.warning(f"Failed to upload to datastore: {str(e)[:200]}")
            logger.info("Falling back to creating dataset from local file...")
            upload_to_datastore = False  # Switch to local file mode
    
    # Create dataset from local file (default or fallback)
    # Note: Azure ML SDK v1 requires files to be in a datastore, so we must upload
    # The upload_to_datastore flag controls whether we use a timestamped path or a simple path
    if not upload_to_datastore:
        logger.info("Creating dataset from local file (upload_to_datastore=false)")
        logger.info("Note: Azure ML requires files in datastore, uploading to simple path without timestamp...")
        datastore = ws.get_default_datastore()
        filename = Path(data_path).name
        target_path = 'data/local/'  # Simple path without timestamp
        
        try:
            # Upload to datastore with simple path
            datastore.upload_files([data_path], target_path=target_path, overwrite=True)
            logger.info(f"File uploaded to datastore path: {target_path}{filename}")
            
            # Create dataset from datastore
            if file_extension in ['.csv', '.tsv']:
                dataset = Dataset.Tabular.from_delimited_files(
                    path=[(datastore, target_path + filename)]
                )
            elif file_extension == '.parquet':
                dataset = Dataset.Tabular.from_parquet_files(
                    path=[(datastore, target_path + filename)]
                )
            else:
                raise ValueError(f"Unsupported file format for dataset registration: {file_extension}")
        except Exception as e:
            logger.error(f"Failed to upload file to datastore: {str(e)}")
            raise RuntimeError(
                "Unable to create dataset. Azure ML SDK v1 requires files to be in a datastore. "
                "Please ensure storage authentication is configured for your workspace's default datastore."
            ) from e
    
    # Register the dataset
    dataset = dataset.register(
        workspace=ws,
        name=dataset_name,
        description=dataset_description,
        create_new_version=True
    )
    
    logger.info(f"Dataset registered: {dataset_name}")
    return dataset


def create_automl_config(
    training_data,  # Can be Dataset or dict with reference info
    config: dict,
    logger: logging.Logger
) -> AutoMLConfig:
    """
    Create AutoML configuration based on config file
    
    Args:
        training_data: Azure ML Dataset for training (or dict with dataset reference info)
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        AutoMLConfig object
    """
    automl_cfg = config['automl']
    data_config = config['data']
    compute_config = config['compute']
    
    # Handle dictionary reference (when local metadata download failed)
    if isinstance(training_data, dict):
        logger.info("Using dataset reference (will be resolved on Azure compute)")
        ws = training_data['workspace']
        dataset_name = training_data['dataset_name']
        dataset_version = training_data.get('dataset_version', 'latest')
        
        # Get dataset on Azure compute by constructing a proper reference
        # The AutoML job will access this using the compute cluster's managed identity
        try:
            if dataset_version == 'latest':
                training_data = Dataset.get_by_name(ws, name=dataset_name)
            else:
                training_data = Dataset.get_by_name(ws, name=dataset_name, version=int(dataset_version))
            logger.info("✓ Dataset reference created for Azure compute")
        except:
            logger.warning("Could not create full dataset object, using name-based reference")
            # Last resort: use dataset name string (AutoML may resolve it)
            training_data = f"azureml:{dataset_name}:{dataset_version}"
    
    # Build base config parameters
    config_params = {
        'task': automl_cfg['task'],
        'training_data': training_data,
        'label_column_name': data_config['label_column'],
        'primary_metric': automl_cfg['primary_metric'],
        'compute_target': compute_config['cluster_name'],
        'experiment_timeout_minutes': automl_cfg['training']['experiment_timeout_minutes'],
        'iteration_timeout_minutes': automl_cfg['training']['iteration_timeout_minutes'],
        'max_concurrent_iterations': automl_cfg['training']['max_concurrent_iterations'],
        'max_cores_per_iteration': automl_cfg['training']['max_cores_per_iteration'],
        'enable_early_stopping': automl_cfg['training']['enable_early_stopping'],
        'verbosity': logging.INFO
    }
    
    # Add cross-validation or validation size
    if automl_cfg['training'].get('n_cross_validations'):
        config_params['n_cross_validations'] = automl_cfg['training']['n_cross_validations']
    elif automl_cfg['training'].get('validation_size'):
        config_params['validation_size'] = automl_cfg['training']['validation_size']
    
    # Configure featurization
    featurization_mode = automl_cfg['featurization']['mode']
    if featurization_mode == 'custom':
        featurization_config = FeaturizationConfig()
        
        # Set blocked transformers
        blocked = automl_cfg['featurization'].get('blocked_transformers', [])
        if blocked:
            featurization_config.blocked_transformers = blocked
        
        # Set column purposes
        col_purposes = automl_cfg['featurization'].get('column_purposes', {})
        if col_purposes:
            featurization_config.column_purposes = col_purposes
        
        # Set drop columns
        drop_cols = automl_cfg['featurization'].get('drop_columns', [])
        if drop_cols:
            featurization_config.drop_columns = drop_cols
        
        # Set transformer params
        transformer_params = automl_cfg['featurization'].get('transformer_params', {})
        if transformer_params:
            featurization_config.transformer_params = transformer_params
        
        config_params['featurization'] = featurization_config
    else:
        config_params['featurization'] = featurization_mode
    
    # Add model allowlist/blocklist
    if automl_cfg['models'].get('allowed'):
        config_params['allowed_models'] = automl_cfg['models']['allowed']
    if automl_cfg['models'].get('blocked'):
        config_params['blocked_models'] = automl_cfg['models']['blocked']
    
    # Add ensemble settings
    if 'ensemble' in automl_cfg:
        config_params['stack_meta_learner_kwargs'] = automl_cfg['ensemble'].get('stack_ensemble_iterations', 5)
        config_params['ensemble_iterations'] = automl_cfg['ensemble'].get('ensemble_iterations', 5)
    
    # ONNX settings
    if automl_cfg['onnx'].get('enable_onnx_compatible_models', False):
        config_params['enable_onnx_compatible_models'] = True
    
    automl_config = AutoMLConfig(**config_params)
    
    logger.info("AutoML configuration created")
    logger.info(f"Task: {automl_cfg['task']}, Primary Metric: {automl_cfg['primary_metric']}")
    return automl_config


def submit_experiment(
    ws: Workspace,
    automl_config: AutoMLConfig,
    config: dict,
    logger: logging.Logger
):
    """
    Submit AutoML experiment
    
    Args:
        ws: Azure ML Workspace
        automl_config: AutoML configuration
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Submitted run object
    """
    experiment_name = config['experiment']['name']
    experiment = Experiment(ws, experiment_name)
    logger.info(f"Submitting experiment: {experiment_name}")
    
    run = experiment.submit(automl_config, show_output=True)
    logger.info(f"Run submitted. Run ID: {run.id}")
    logger.info(f"Monitor at: {run.get_portal_url()}")
    
    return run


def register_best_model(
    run,
    config: dict,
    logger: logging.Logger
) -> Model:
    """
    Register the best model from the run
    
    Args:
        run: Completed AutoML run
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Registered model
    """
    output_config = config.get('output', {})
    if not output_config.get('register_model', True):
        logger.info("Model registration skipped")
        return None
    
    model_name = output_config.get('model_name', 'automl_model')
    model_tags = output_config.get('model_tags', {})
    
    best_run, fitted_model = run.get_output()
    
    # Register model
    model = best_run.register_model(
        model_name=model_name,
        tags=model_tags
    )
    
    logger.info(f"Model registered: {model_name} (version {model.version})")
    return model


def main():
    """
    Main execution function
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Azure ML AutoML Training Script')
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
    logger.info("="*60)
    logger.info("Azure ML AutoML Training")
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info("="*60)
    
    try:
        # Connect to workspace using environment variables
        logger.info("Connecting to Azure ML Workspace")
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_ML_WORKSPACE")
        
        if not all([subscription_id, resource_group, workspace_name]):
            raise ValueError("Missing Azure credentials in .env file")
        
        # Get authentication method from config
        auth_config = config.get('authentication', {})
        auth_method = auth_config.get('method', 'interactive').lower()
        
        logger.info(f"Authentication method: {auth_method}")
        
        # Configure authentication based on method
        auth = None
        if auth_method == 'msi' or auth_method == 'managed_identity':
            logger.info("Using System-Assigned Managed Identity (MSI) authentication")
            auth = MsiAuthentication()
        elif auth_method == 'uami' or auth_method == 'user_assigned_managed_identity':
            # User-Assigned Managed Identity
            client_id = auth_config.get('managed_identity_client_id')
            if not client_id:
                raise ValueError(
                    "authentication.managed_identity_client_id is required when using method='uami'. "
                    "Please add the client ID of your user-assigned managed identity to config.yaml"
                )
            logger.info(f"Using User-Assigned Managed Identity (UAMI) authentication")
            logger.info(f"  Client ID: {client_id}")
            logger.info("  NOTE: UAMI only works when running on Azure resources (VMs, Container Instances, etc.)")
            logger.info("        For local testing, use method='interactive' or 'cli' instead")
            
            try:
                auth = MsiAuthentication(client_id=client_id)
            except Exception as e:
                logger.error(f"Failed to initialize UAMI authentication: {e}")
                logger.error("If running locally, UAMI will not work. Switch to 'interactive' or 'cli' method.")
                raise
        elif auth_method == 'cli':
            logger.info("Using Azure CLI authentication")
            # Azure CLI authentication is implicit when auth=None
            auth = None
        else:  # interactive (default)
            logger.info("Using Interactive Browser authentication")
            auth = InteractiveLoginAuthentication()
        
        # Get workspace with configured authentication
        ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=auth
        )
        logger.info(f"Connected to workspace: {ws.name}")
        logger.info("Note: Storage operations will use the same authentication method")
        
        # Get or create dataset
        data_config = config['data']
        use_existing_dataset = data_config.get('use_existing_dataset', False)
        
        if use_existing_dataset:
            # Use existing dataset (no preprocessing needed)
            dataset = get_or_create_dataset(ws, None, config, logger)
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
                logger.info(f"Preprocessed data saved to: {preprocessed_path}")
            else:
                # Create temporary file for upload
                preprocessed_path = f"temp_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(preprocessed_path, index=False)
            
            # Upload and register dataset
            dataset = get_or_create_dataset(ws, str(preprocessed_path), config, logger)
        
        # Create AutoML configuration
        automl_config = create_automl_config(dataset, config, logger)
        
        # Submit experiment
        run = submit_experiment(ws, automl_config, config, logger)
        
        # Wait for completion
        logger.info("Waiting for run to complete...")
        logger.info("You can close this script and monitor progress in Azure ML Studio")
        run.wait_for_completion(show_output=True)
        
        # Get best model and metrics
        best_run, fitted_model = run.get_output()
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info(f"Best run ID: {best_run.id}")
        logger.info(f"Best model metrics:")
        for metric, value in best_run.get_metrics().items():
            logger.info(f"  {metric}: {value}")
        logger.info("="*60)
        
        # Register model
        model = register_best_model(run, config, logger)
        
        # Clean up temporary file if created
        if not output_config.get('save_preprocessed_data', True):
            if Path(preprocessed_path).exists():
                Path(preprocessed_path).unlink()
                logger.info("Temporary preprocessed data file removed")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
