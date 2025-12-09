"""
Submit RAI Dashboard job for sklearn model.

This uses the standard sklearn model that IS compatible with RAI dashboard,
unlike AutoML models which embed internal Azure ML classes that break RAI.
"""

import argparse
import logging
import uuid
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import PipelineJob
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_rai_pipeline(
    ml_client: MLClient,
    credential,
    model_name: str,
    model_version: str,
    train_data_name: str,
    train_data_version: str,
    test_data_name: str,
    test_data_version: str,
    target_column: str,
    compute_name: str,
):
    """Create the RAI pipeline job."""
    
    # Get RAI components from registry using a registry client
    registry_name = "azureml"
    
    # Create a registry client to access RAI components
    registry_ml_client = MLClient(
        credential=credential,
        registry_name=registry_name,
    )
    
    # Get the RAI components (using correct names: rai_tabular_*)
    rai_constructor = registry_ml_client.components.get(
        name="rai_tabular_insight_constructor",
        label="latest",
    )
    
    rai_explanation = registry_ml_client.components.get(
        name="rai_tabular_explanation",
        label="latest",
    )
    
    rai_erroranalysis = registry_ml_client.components.get(
        name="rai_tabular_erroranalysis",
        label="latest",
    )
    
    rai_gather = registry_ml_client.components.get(
        name="rai_tabular_insight_gather",
        label="latest",
    )
    
    logger.info(f"Loaded RAI components from registry")
    
    # Build the pipeline using the registry components
    model_id = f"azureml:{model_name}:{model_version}"
    
    @pipeline(
        compute=compute_name,
        description="RAI Dashboard for sklearn CVD model",
    )
    def rai_pipeline(target_column_name, train_data, test_data):
        # Constructor - creates the RAI insights object
        # use_model_dependency=True tells RAI to use the model's conda.yaml for dependencies
        constructor = rai_constructor(
            title="sklearn-cvd-rai-dashboard",
            task_type="classification",
            model_info=model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
            use_model_dependency=True,  # Use model's environment for Python/NumPy compatibility
        )
        
        # Add error analysis component
        erroranalysis = rai_erroranalysis(
            rai_insights_dashboard=constructor.outputs.rai_insights_dashboard,
        )
        
        # Add explanation component
        explanation = rai_explanation(
            rai_insights_dashboard=constructor.outputs.rai_insights_dashboard,
            comment="SHAP explanations for sklearn model",
        )
        
        # Gather all insights
        gather = rai_gather(
            constructor=constructor.outputs.rai_insights_dashboard,
            insight_3=erroranalysis.outputs.error_analysis,
            insight_4=explanation.outputs.explanation,
        )
        gather.outputs.dashboard.mode = "upload"
        
        return {"dashboard": gather.outputs.dashboard}
    
    # Create pipeline inputs - use mltable for the data
    train_data_path = f"azureml:{train_data_name}:{train_data_version}"
    test_data_path = f"azureml:{test_data_name}:{test_data_version}"
    
    # RAI components require mltable format
    train_input = Input(type="mltable", path=train_data_path, mode="download")
    test_input = Input(type="mltable", path=test_data_path, mode="download")
    
    pipeline_job = rai_pipeline(
        target_column_name=target_column,
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
    parser = argparse.ArgumentParser(description="Submit RAI Dashboard job for sklearn model")
    parser.add_argument("--model-name", default="sklearn-cvd-model", help="Model name in Azure ML")
    parser.add_argument("--model-version", default="2", help="Model version")
    parser.add_argument("--train-data-name", default="sklearn-cvd-train-mltable", help="Training data asset name")
    parser.add_argument("--train-data-version", default="1", help="Training data version")
    parser.add_argument("--test-data-name", default="sklearn-cvd-test-mltable", help="Test data asset name")
    parser.add_argument("--test-data-version", default="1", help="Test data version")
    parser.add_argument("--target-column", default="MACE", help="Target column name")
    parser.add_argument("--compute", default="automl", help="Compute cluster name")
    parser.add_argument("--subscription-id", default="1c47c29b-10d8-4bc6-a024-05ec921662cb")
    parser.add_argument("--resource-group", default="rg-upmc")
    parser.add_argument("--workspace", default="mlw-25")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Submitting RAI Dashboard Job for Sklearn Model")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}:{args.model_version}")
    logger.info(f"Train Data: {args.train_data_name}:{args.train_data_version}")
    logger.info(f"Test Data: {args.test_data_name}:{args.test_data_version}")
    logger.info(f"Target Column: {args.target_column}")
    logger.info(f"Compute: {args.compute}")
    
    # Create ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )
    logger.info(f"Connected to workspace: {args.workspace}")
    
    # Create and submit pipeline
    pipeline_job = get_rai_pipeline(
        ml_client=ml_client,
        credential=credential,
        model_name=args.model_name,
        model_version=args.model_version,
        train_data_name=args.train_data_name,
        train_data_version=args.train_data_version,
        test_data_name=args.test_data_name,
        test_data_version=args.test_data_version,
        target_column=args.target_column,
        compute_name=args.compute,
    )
    
    # Submit the job
    logger.info("Submitting RAI pipeline job...")
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    
    logger.info("=" * 70)
    logger.info("JOB SUBMITTED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Job Name: {submitted_job.name}")
    logger.info(f"Job Status: {submitted_job.status}")
    logger.info(f"Job URL: {submitted_job.studio_url}")
    
    return submitted_job


if __name__ == "__main__":
    main()
