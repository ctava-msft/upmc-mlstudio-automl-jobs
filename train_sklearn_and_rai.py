"""
Train a Pure Sklearn Model for RAI Dashboard

This script trains a simple sklearn classifier on the same dataset,
which will be fully compatible with RAI dashboard components.

Since AutoML models cannot be loaded locally due to internal dependencies,
this provides an alternative path to get RAI insights.
"""

import os
import sys
import argparse
import logging
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential

load_dotenv()

TARGET_COLUMN = 'MACE'


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_data(data_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load data from CSV or parquet file."""
    logger.info(f"Loading data from {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def preprocess_data(df: pd.DataFrame, logger: logging.Logger):
    """Preprocess data for sklearn model."""
    logger.info("Preprocessing data...")
    
    # Make a copy
    df = df.copy()
    
    # Ensure target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    # Separate features and target
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN])
    
    # Handle categorical columns - encode them
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle NaN values
        X[col] = X[col].fillna('MISSING')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Handle numeric NaN values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    
    # Convert to float32
    X = X.astype(np.float32)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
    
    return X, y, label_encoders


def train_model(X, y, logger: logging.Logger):
    """Train a GradientBoosting classifier."""
    logger.info("Training GradientBoostingClassifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Train accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Store feature names
    model.feature_names_in_ = np.array(X.columns.tolist())
    
    return model, X_train, X_test, y_train, y_test


def save_mlflow_model(model, output_path: str, X_sample: pd.DataFrame, logger: logging.Logger):
    """Save model in MLflow format."""
    logger.info(f"Saving MLflow model to {output_path}")
    
    # Create signature from sample data
    signature = mlflow.models.infer_signature(X_sample, model.predict(X_sample))
    
    # Save model
    mlflow.sklearn.save_model(
        model,
        output_path,
        signature=signature,
    )
    
    logger.info("Model saved successfully")
    return output_path


def register_model(ml_client: MLClient, model_path: str, model_name: str, logger: logging.Logger) -> Model:
    """Register the model in Azure ML."""
    logger.info(f"Registering model '{model_name}'")
    
    model = Model(
        name=model_name,
        path=model_path,
        type=AssetTypes.MLFLOW_MODEL,
        description="Sklearn GradientBoosting model for Secondary CVD Risk - RAI compatible",
    )
    
    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Model registered: {model_name}:{registered_model.version}")
    
    return registered_model


def get_rai_components(ml_client_registry: MLClient, logger: logging.Logger) -> dict:
    """Get RAI components from registry."""
    logger.info("Fetching RAI components...")
    
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
    """Create full RAI dashboard pipeline."""
    
    model_id = f"{model.name}:{model.version}"
    azureml_model_id = f"azureml:{model_id}"
    train_data_id = f"azureml:{train_data.name}:{train_data.version}"
    test_data_id = f"azureml:{test_data.name}:{test_data.version}"
    
    logger.info("Creating RAI pipeline:")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Train data: {train_data_id}")
    logger.info(f"  Test data: {test_data_id}")
    
    rai_constructor = rai_components['constructor']
    rai_erroranalysis = rai_components['erroranalysis']
    rai_explanation = rai_components['explanation']
    rai_gather = rai_components['gather']
    
    @dsl.pipeline(
        compute=compute_name,
        description="Full RAI Dashboard for Secondary CVD Risk sklearn model",
        experiment_name=f"RAI_dashboard_{model.name}",
    )
    def rai_full_pipeline(target_column_name, train_data, test_data):
        # Constructor
        create_rai_job = rai_constructor(
            title="RAI Dashboard - Secondary CVD Risk (sklearn)",
            task_type="classification",
            model_info=model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
        )
        create_rai_job.set_limits(timeout=3600)
        
        # Error Analysis
        error_job = rai_erroranalysis(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        error_job.set_limits(timeout=3600)
        
        # Explanations
        explanation_job = rai_explanation(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            comment="SHAP feature explanations",
        )
        explanation_job.set_limits(timeout=3600)
        
        # Gather
        rai_gather_job = rai_gather(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_3=error_job.outputs.error_analysis,
            insight_4=explanation_job.outputs.explanation,
        )
        rai_gather_job.set_limits(timeout=3600)
        rai_gather_job.outputs.dashboard.mode = "upload"
        
        return {"dashboard": rai_gather_job.outputs.dashboard}
    
    # Create inputs
    train_input = Input(type="mltable", path=train_data_id, mode="download")
    test_input = Input(type="mltable", path=test_data_id, mode="download")
    
    pipeline_job = rai_full_pipeline(
        target_column_name=TARGET_COLUMN,
        train_data=train_input,
        test_data=test_input,
    )
    
    rand_path = str(uuid.uuid4())
    pipeline_job.outputs.dashboard = Output(
        path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
        mode="upload",
        type="uri_folder",
    )
    
    return pipeline_job


def main():
    parser = argparse.ArgumentParser(description="Train sklearn model and run RAI dashboard")
    parser.add_argument("--data-path", default="./data/secondary_cvd_risk_min/secondary-cvd-risk.csv",
                        help="Path to training data (CSV or parquet)")
    parser.add_argument("--model-name", default="secondary-cvd-risk-sklearn",
                        help="Name for registered model")
    parser.add_argument("--model-output", default="./models/sklearn_rai_model",
                        help="Local path to save model")
    parser.add_argument("--compute", default="aml-cluster",
                        help="Compute cluster for RAI job")
    parser.add_argument("--data-name", default="secondary-cvd-risk",
                        help="Azure ML data asset name")
    parser.add_argument("--data-version", default="1",
                        help="Azure ML data asset version")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing model")
    parser.add_argument("--skip-rai", action="store_true",
                        help="Skip RAI job submission")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for RAI job to complete")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 70)
    logger.info("Train Sklearn Model and Run RAI Dashboard")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Connect to Azure ML
    subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = os.getenv('AZURE_ML_WORKSPACE')
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Missing Azure ML environment variables")
    
    credential = AzureCliCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    logger.info(f"Connected to workspace: {workspace_name}")
    
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        registry_name="azureml",
    )
    
    registered_model = None
    
    if not args.skip_train:
        # Load and preprocess data
        data_path = os.path.abspath(args.data_path)
        df = load_data(data_path, logger)
        X, y, encoders = preprocess_data(df, logger)
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_model(X, y, logger)
        
        # Save model
        model_output = os.path.abspath(args.model_output)
        if os.path.exists(model_output):
            import shutil
            shutil.rmtree(model_output)
        save_mlflow_model(model, model_output, X_test.head(10), logger)
        
        # Register model
        registered_model = register_model(ml_client, model_output, args.model_name, logger)
    else:
        logger.info(f"Using existing model: {args.model_name}")
        models = list(ml_client.models.list(name=args.model_name))
        if models:
            registered_model = models[0]
            logger.info(f"Found model: {args.model_name}:{registered_model.version}")
        else:
            raise ValueError(f"Model '{args.model_name}' not found")
    
    if args.skip_rai:
        logger.info("Skipping RAI job (--skip-rai)")
        return 0
    
    # Get data assets
    train_data = ml_client.data.get(name=args.data_name, version=args.data_version)
    test_data = ml_client.data.get(name=args.data_name, version=args.data_version)
    
    # Get RAI components
    rai_components = get_rai_components(ml_client_registry, logger)
    
    # Create and submit pipeline
    pipeline_job = create_rai_pipeline(
        rai_components=rai_components,
        model=registered_model,
        train_data=train_data,
        test_data=test_data,
        compute_name=args.compute,
        logger=logger
    )
    
    logger.info("Submitting RAI dashboard job...")
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("RAI Dashboard Job Submitted")
    logger.info("=" * 70)
    logger.info(f"Job Name: {created_job.name}")
    logger.info(f"Status: {created_job.status}")
    logger.info(f"Studio URL: {created_job.studio_url}")
    logger.info("=" * 70)
    
    if args.wait:
        logger.info("Waiting for job completion...")
        terminal_states = ["Completed", "Failed", "Canceled"]
        
        while created_job.status not in terminal_states:
            time.sleep(30)
            created_job = ml_client.jobs.get(created_job.name)
            logger.info(f"Status: {created_job.status}")
        
        if created_job.status == "Completed":
            logger.info("✅ RAI dashboard job completed!")
            logger.info(f"View dashboard: {created_job.studio_url}")
        else:
            logger.error(f"❌ Job ended with status: {created_job.status}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
