"""
Azure ML AutoML Classification Job

This script runs an AutoML classification job on the secondary-cvd-risk dataset.
The resulting model will be compatible with RAI dashboards.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

from azure.ai.ml import MLClient, automl, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.automl import ClassificationPrimaryMetrics
from azure.identity import AzureCliCredential

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


def main():
    parser = argparse.ArgumentParser(description="Run AutoML Classification Job")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument("--timeout-minutes", type=int, default=60, help="Training timeout in minutes")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 70)
    logger.info("Azure ML AutoML Classification Job")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Get workspace details from environment
    subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = os.getenv('AZURE_ML_WORKSPACE')
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Missing AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, or AZURE_ML_WORKSPACE")
    
    # Connect to workspace
    credential = AzureCliCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    logger.info(f"Connected to workspace: {workspace_name}")
    
    # Get the existing data asset
    data_name = "secondary-cvd-risk"
    data_version = "1"
    
    logger.info(f"Using data asset: {data_name}:{data_version}")
    
    # Create training data input
    training_data = Input(
        type=AssetTypes.MLTABLE,
        path=f"azureml:{data_name}:{data_version}"
    )
    
    # Configure AutoML classification job
    classification_job = automl.classification(
        compute="automl",
        experiment_name="secondary-cvd-risk-automl",
        training_data=training_data,
        target_column_name="MACE",
        primary_metric=ClassificationPrimaryMetrics.AUC_WEIGHTED,
        n_cross_validations=5,
        enable_model_explainability=True,  # Enable for RAI
    )
    
    # Set job name
    classification_job.display_name = "secondary-cvd-risk-classification"
    classification_job.name = f"secondary-cvd-risk-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Set limits
    trial_timeout = min(10, args.timeout_minutes - 1)  # Trial must be less than experiment timeout
    classification_job.set_limits(
        timeout_minutes=args.timeout_minutes,
        trial_timeout_minutes=trial_timeout,
        max_trials=1,
        max_concurrent_trials=1,
        enable_early_termination=True,
    )
    
    # Set training settings
    classification_job.set_training(
        enable_onnx_compatible_models=True,  # ONNX models work better with RAI
        enable_stack_ensemble=False,  # Simpler models for RAI compatibility
        enable_vote_ensemble=False,
    )
    
    # Set featurization
    classification_job.set_featurization(mode="auto")
    
    logger.info("AutoML job configuration:")
    logger.info(f"  Compute: automl")
    logger.info(f"  Target column: MACE")
    logger.info(f"  Primary metric: AUC_WEIGHTED")
    logger.info(f"  Timeout: {args.timeout_minutes} minutes")
    logger.info(f"  Max trials: 10")
    
    # Submit the job
    logger.info("Submitting AutoML job...")
    returned_job = ml_client.jobs.create_or_update(classification_job)
    
    logger.info(f"Job submitted: {returned_job.name}")
    logger.info(f"Studio URL: {returned_job.studio_url}")
    
    if args.wait:
        logger.info("Waiting for job completion...")
        ml_client.jobs.stream(returned_job.name)
        
        # Get final status
        final_job = ml_client.jobs.get(returned_job.name)
        logger.info(f"Final status: {final_job.status}")
        
        if final_job.status == "Completed":
            logger.info("✅ AutoML job completed successfully!")
            logger.info("The best model has been registered. You can now run the RAI dashboard.")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Find the registered model name from the job outputs in Azure ML Studio")
            logger.info("2. Update rai_config.yaml with the model name")
            logger.info("3. Run: python submit_rai_job.py")
        else:
            logger.error(f"❌ AutoML job ended with status: {final_job.status}")
            return 1
    else:
        logger.info("Job submitted. Use --wait to monitor progress.")
        logger.info("Or view progress in Azure ML Studio.")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("AutoML Job Summary")
    logger.info("=" * 70)
    logger.info(f"Job Name: {returned_job.name}")
    logger.info(f"Status: {returned_job.status}")
    logger.info(f"Studio URL: {returned_job.studio_url}")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
