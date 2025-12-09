"""
Train a Standard Sklearn Model for RAI Dashboard

This script trains a standard scikit-learn model that IS compatible with RAI Dashboard.
The key difference from AutoML models is that sklearn models don't embed internal
Azure ML SDK classes that cause deserialization errors.

Usage:
    python train_sklearn_for_rai.py --data-path data/secondary_cvd_risk_min/secondary-cvd-risk.csv
"""

import os
import sys
import argparse
import logging
import pickle
import json
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn

# Configuration
TARGET_COLUMN = 'MACE'
RANDOM_STATE = 42


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str, logger: logging.Logger):
    """Load and preprocess the data for training."""
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        # Try alternative names
        alt_names = ['CVD_RISK', 'target', 'label', 'y']
        for alt in alt_names:
            if alt in df.columns:
                df = df.rename(columns={alt: TARGET_COLUMN})
                logger.info(f"Renamed column '{alt}' to '{TARGET_COLUMN}'")
                break
        else:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Available: {list(df.columns)}")
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns
    
    # Fill numeric missing with median
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('UNKNOWN')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, list(X.columns), label_encoders


def train_model(X, y, model_type: str, logger: logging.Logger):
    """Train a sklearn model."""
    logger.info(f"Training {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Select model
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == "logistic_regression":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_test, y_proba)
    
    logger.info(f"Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, metrics, (X_train, X_test, y_train, y_test)


def save_mlflow_model(model, feature_names: list, output_path: str, logger: logging.Logger):
    """Save model in MLflow format compatible with RAI."""
    logger.info(f"Saving MLflow model to {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save model using sklearn flavor
    mlflow.sklearn.save_model(
        sk_model=model,
        path=output_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
    )
    
    # Also save feature names for reference
    feature_names_path = os.path.join(output_path, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    logger.info(f"Model saved successfully")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Train sklearn model for RAI')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--model-type', type=str, default='gradient_boosting',
                        choices=['gradient_boosting', 'random_forest', 'logistic_regression'],
                        help='Type of model to train')
    parser.add_argument('--output-path', type=str, default='models/sklearn_rai_model',
                        help='Output path for the model')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level')
    
    args = parser.parse_args()
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 70)
    logger.info("Training Sklearn Model for RAI Dashboard")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output path: {args.output_path}")
    
    try:
        # Load and preprocess data
        X, y, feature_names, encoders = load_and_preprocess_data(args.data_path, logger)
        
        # Train model
        model, metrics, splits = train_model(X, y, args.model_type, logger)
        
        # Save model
        save_mlflow_model(model, feature_names, args.output_path, logger)
        
        # Save test data for RAI
        X_train, X_test, y_train, y_test = splits
        test_data = X_test.copy()
        test_data[TARGET_COLUMN] = y_test.values
        
        test_data_path = os.path.join(os.path.dirname(args.output_path), "test_data.csv")
        test_data.to_csv(test_data_path, index=False)
        logger.info(f"Test data saved to {test_data_path}")
        
        train_data = X_train.copy()
        train_data[TARGET_COLUMN] = y_train.values
        train_data_path = os.path.join(os.path.dirname(args.output_path), "train_data.csv")
        train_data.to_csv(train_data_path, index=False)
        logger.info(f"Train data saved to {train_data_path}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Model trained and saved successfully!")
        logger.info("=" * 70)
        logger.info(f"Model path: {args.output_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Register the model in Azure ML:")
        logger.info(f"   az ml model create --name sklearn-cvd-model --path {args.output_path} --type mlflow_model")
        logger.info("")
        logger.info("2. Create data assets:")
        logger.info(f"   az ml data create --name sklearn-train-data --path {train_data_path} --type uri_file")
        logger.info(f"   az ml data create --name sklearn-test-data --path {test_data_path} --type uri_file")
        logger.info("")
        logger.info("3. Run RAI dashboard (this model IS compatible with RAI!)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
