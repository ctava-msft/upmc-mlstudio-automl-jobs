"""
Train a simple sklearn model for RAI dashboard demonstration.

Since the AutoML pickle model has dependencies that can't be resolved 
in MLflow/RAI environments, we'll train a fresh sklearn model that:
1. Works with the same secondary-cvd-risk dataset
2. Is fully compatible with MLflow and RAI components
3. Demonstrates the RAI dashboard capabilities
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn


def train_and_save_model():
    """Train a sklearn model and save as MLflow format"""
    
    # Load the data
    data_path = "data/secondary_cvd_risk_min/secondary-cvd-risk.csv"
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}... (total {len(df.columns)})")
    
    # Target column
    target_column = "MACE"
    print(f"\nTarget column: {target_column}")
    print(f"Target distribution:\n{df[target_column].value_counts()}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical columns - convert to numeric
    print("\nProcessing categorical columns...")
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")
    
    # Simple encoding for categorical variables
    for col in categorical_cols:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        else:
            # Label encode
            X[col] = pd.factorize(X[col])[0]
    
    # Convert boolean target to int
    if y.dtype == 'bool':
        y = y.astype(int)
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Feature dtypes:\n{X.dtypes.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train a GradientBoostingClassifier (similar to what AutoML might choose)
    print("\nTraining GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save as MLflow model
    output_path = "models/secondary_cvd_risk_mlflow"
    
    # Clean up existing
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    
    print(f"\nSaving MLflow model to {output_path}...")
    
    # Define signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Save the model
    mlflow.sklearn.save_model(
        sk_model=model,
        path=output_path,
        signature=signature,
        input_example=X_train.head(5),
    )
    
    print(f"MLflow model saved successfully!")
    print(f"\nContents of {output_path}:")
    for item in os.listdir(output_path):
        print(f"  - {item}")
    
    # Verify load works
    print("\nVerifying model can be loaded...")
    loaded = mlflow.sklearn.load_model(output_path)
    test_pred = loaded.predict(X_test.head(1))
    print(f"Test prediction: {test_pred}")
    print("✓ Model verified!")
    
    # Save the feature names for reference
    feature_names = X.columns.tolist()
    with open(os.path.join(output_path, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    print(f"Saved {len(feature_names)} feature names")
    
    return output_path


if __name__ == "__main__":
    train_and_save_model()
