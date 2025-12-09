"""
Train a simple scikit-learn model for RAI compatibility.
This avoids the AutoML SDK dependencies that cause issues with RAI components.
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import yaml

load_dotenv()


def load_config():
    """Load configuration from rai_config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), "rai_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_ml_client():
    """Create and return an ML client"""
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    return ml_client


def load_data(ml_client, data_asset_name, data_asset_version, target_column):
    """Load data from Azure ML data asset"""
    print(f"Loading data asset: {data_asset_name}:{data_asset_version}")
    
    # Get the data asset
    data_asset = ml_client.data.get(name=data_asset_name, version=data_asset_version)
    
    # Read the CSV file
    csv_path = os.path.join(data_asset.path.replace("azureml://", ""), "*.csv")
    
    # For local development, try to find the local data
    local_data_dir = os.path.join(os.path.dirname(__file__), "data", "secondary_cvd_risk_min")
    csv_files = [f for f in os.listdir(local_data_dir) if f.endswith('.csv')]
    
    if csv_files:
        csv_path = os.path.join(local_data_dir, csv_files[0])
        print(f"Loading local data from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No CSV files found in {local_data_dir}")
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Target column: {target_column}")
    print(f"Target distribution:\n{df[target_column].value_counts()}")
    
    return df


def prepare_features(df, target_column):
    """Prepare features for training"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert boolean target to int if needed
    if y.dtype == bool:
        y = y.astype(int)
    elif y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Convert boolean columns to int
    for col in categorical_cols:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
            numeric_cols.append(col)
            categorical_cols.remove(col)
    
    # Handle remaining categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, numeric_cols, categorical_cols


def train_model(X_train, y_train, model_type="random_forest"):
    """Train a model"""
    print(f"Training {model_type} model...")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == "logistic_regression":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
    }
    
    if y_proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y_test, y_proba)
        except:
            pass
    
    print("\nModel Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return metrics


def register_model(ml_client, model, X_train, model_name, metrics):
    """Register model to Azure ML using MLflow"""
    print(f"\nRegistering model: {model_name}")
    
    # Set MLflow tracking URI
    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment
    experiment_name = "secondary-cvd-risk-simple"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name) as run:
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # Log model with signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log the model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=X_train.head(5)
        )
        
        print(f"Model registered successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model name: {model_name}")
        
    return model_name


def main():
    parser = argparse.ArgumentParser(description="Train a simple sklearn model for RAI compatibility")
    parser.add_argument("--model-type", type=str, default="random_forest", 
                        choices=["random_forest", "gradient_boosting", "logistic_regression"],
                        help="Type of model to train")
    parser.add_argument("--model-name", type=str, default="secondary-cvd-risk-simple",
                        help="Name to register the model with")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of data for testing")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    target_column = config.get("target_column", "MACE")
    data_asset_name = config.get("data_asset", {}).get("name", "secondary-cvd-risk")
    data_asset_version = config.get("data_asset", {}).get("version", "1")
    
    # Create ML client
    ml_client = get_ml_client()
    
    # Load data
    df = load_data(ml_client, data_asset_name, data_asset_version, target_column)
    
    # Prepare features
    X, y, numeric_cols, categorical_cols = prepare_features(df, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, args.model_type)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Register model
    model_name = register_model(ml_client, model, X_train, args.model_name, metrics)
    
    print(f"\n{'='*50}")
    print(f"Model training complete!")
    print(f"Model name: {model_name}")
    print(f"Update rai_config.yaml with this model name to use with RAI")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
