#!/usr/bin/env python
"""
SHAP Explanations Template for Machine Learning Models
=======================================================

This template provides a reusable framework for generating SHAP (SHapley Additive exPlanations)
explanations for any classification or regression model.

Features:
---------
- Support for both classification and regression tasks
- Multiple model types (tree-based, linear, neural networks)
- Configurable output directory and file naming
- Interactive and static visualizations
- Export capabilities for reports and dashboards

Usage:
------
1. Direct execution with demo data:
   python shap_demo.py

2. With your own data and model:
   python shap_demo.py --data-path your_data.csv --model-path your_model.pkl

3. As a module:
   from shap_demo import SHAPExplainer
   explainer = SHAPExplainer(model, X_train, feature_names)
   explainer.generate_all_explanations(X_test, output_dir="./explanations")

Configuration:
--------------
Modify the CONFIG dictionary below to customize behavior for your use case.

Author: Your Organization
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION - Modify these settings for your use case
# =============================================================================

CONFIG = {
    # Data settings
    "target_column": "target",          # Name of your target column
    "sample_size": 100,                  # Number of samples for SHAP analysis
    "test_size": 0.2,                    # Train/test split ratio
    "random_state": 42,                  # Random seed for reproducibility
    
    # Model settings
    "task_type": "classification",       # "classification" or "regression"
    "model_type": "tree",                # "tree", "linear", "kernel", or "deep"
    
    # Output settings
    "output_dir": "./explanations",      # Directory for saving outputs
    "save_format": "png",                # Image format: "png", "pdf", "svg"
    "dpi": 300,                          # Image resolution
    "create_html": True,                 # Generate interactive HTML plots
    
    # Visualization settings
    "max_features_display": 20,          # Max features to show in plots
    "color_positive": "#ff0051",         # Color for positive SHAP values
    "color_negative": "#008bfb",         # Color for negative SHAP values
    
    # Logging
    "log_level": "INFO",                 # Logging verbosity
}


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging with timestamp and level formatting."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging(CONFIG["log_level"])


# =============================================================================
# SHAP EXPLAINER CLASS
# =============================================================================

class SHAPExplainer:
    """
    A reusable class for generating SHAP explanations.
    
    This class encapsulates all SHAP functionality and provides a clean API
    for generating various types of explanations and visualizations.
    
    Parameters:
    -----------
    model : Any
        A trained machine learning model with predict/predict_proba methods
    background_data : pd.DataFrame or np.ndarray
        Training data used as background for SHAP calculations
    feature_names : List[str], optional
        Names of features (inferred from DataFrame if not provided)
    task_type : str
        Either "classification" or "regression"
    model_type : str
        Type of model: "tree", "linear", "kernel", or "deep"
    
    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> explainer = SHAPExplainer(model, X_train, feature_names)
    >>> explainer.generate_all_explanations(X_test)
    """
    
    def __init__(
        self,
        model: Any,
        background_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
        model_type: str = "tree"
    ):
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError(
                "SHAP library not installed. Install with: pip install shap"
            )
        
        self.model = model
        self.task_type = task_type
        self.model_type = model_type
        
        # Handle background data
        if isinstance(background_data, pd.DataFrame):
            self.background_data = background_data
            self.feature_names = list(background_data.columns) if feature_names is None else feature_names
        else:
            self.background_data = pd.DataFrame(background_data, columns=feature_names)
            self.feature_names = feature_names or [f"feature_{i}" for i in range(background_data.shape[1])]
        
        # Initialize the appropriate explainer
        self.explainer = self._create_explainer()
        self.shap_values = None
        
        logger.info(f"SHAPExplainer initialized for {task_type} task with {model_type} model")
        logger.info(f"Background data shape: {self.background_data.shape}")
    
    def _create_explainer(self):
        """Create the appropriate SHAP explainer based on model type."""
        if self.model_type == "tree":
            logger.info("Using TreeExplainer (optimized for tree-based models)")
            return self.shap.TreeExplainer(self.model)
        
        elif self.model_type == "linear":
            logger.info("Using LinearExplainer (optimized for linear models)")
            return self.shap.LinearExplainer(self.model, self.background_data)
        
        elif self.model_type == "deep":
            logger.info("Using DeepExplainer (for neural networks)")
            return self.shap.DeepExplainer(self.model, self.background_data.values)
        
        else:  # kernel or unknown
            logger.info("Using KernelExplainer (model-agnostic, slower)")
            # Sample background data for efficiency
            background_sample = self.shap.sample(self.background_data, min(100, len(self.background_data)))
            return self.shap.KernelExplainer(self.model.predict_proba, background_sample)
    
    def calculate_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_size: Optional[int] = None
    ) -> Any:
        """
        Calculate SHAP values for the given data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to explain
        sample_size : int, optional
            Number of samples to use (defaults to CONFIG["sample_size"])
        
        Returns:
        --------
        shap.Explanation
            SHAP values as an Explanation object
        """
        sample_size = sample_size or CONFIG["sample_size"]
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Sample if needed
        if len(X) > sample_size:
            logger.info(f"Sampling {sample_size} from {len(X)} instances")
            X = X.sample(n=sample_size, random_state=CONFIG["random_state"])
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        
        # Calculate SHAP values using the new API
        self.shap_values = self.explainer(X)
        
        # Handle multi-class/binary classification
        if self.task_type == "classification" and len(self.shap_values.values.shape) == 3:
            # For binary classification, use positive class (index 1)
            self.shap_values_display = self.shap_values[:, :, 1]
            logger.info("Using positive class SHAP values for binary classification")
        else:
            self.shap_values_display = self.shap_values
        
        logger.info("SHAP values calculated successfully")
        return self.shap_values
    
    def plot_feature_importance(
        self,
        output_path: Optional[str] = None,
        max_features: int = None,
        title: str = "SHAP Feature Importance"
    ) -> None:
        """
        Create a bar plot showing mean absolute SHAP values per feature.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the plot
        max_features : int, optional
            Maximum number of features to display
        title : str
            Plot title
        """
        if self.shap_values is None:
            raise ValueError("Call calculate_shap_values() first")
        
        max_features = max_features or CONFIG["max_features_display"]
        
        plt.figure(figsize=(10, 6))
        self.shap.plots.bar(
            self.shap_values_display,
            max_display=max_features,
            show=False
        )
        plt.title(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        
        plt.close()
    
    def plot_summary(
        self,
        output_path: Optional[str] = None,
        max_features: int = None,
        title: str = "SHAP Summary Plot"
    ) -> None:
        """
        Create a beeswarm plot showing feature impact distribution.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the plot
        max_features : int, optional
            Maximum number of features to display
        title : str
            Plot title
        """
        if self.shap_values is None:
            raise ValueError("Call calculate_shap_values() first")
        
        max_features = max_features or CONFIG["max_features_display"]
        
        plt.figure(figsize=(10, 8))
        self.shap.plots.beeswarm(
            self.shap_values_display,
            max_display=max_features,
            show=False
        )
        plt.title(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
            logger.info(f"Summary plot saved to {output_path}")
        
        plt.close()
    
    def plot_waterfall(
        self,
        sample_index: int = 0,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Create a waterfall plot for a single prediction.
        
        Parameters:
        -----------
        sample_index : int
            Index of the sample to explain
        output_path : str, optional
            Path to save the plot
        title : str, optional
            Plot title
        """
        if self.shap_values is None:
            raise ValueError("Call calculate_shap_values() first")
        
        title = title or f"SHAP Waterfall Plot (Sample {sample_index})"
        
        plt.figure(figsize=(10, 8))
        self.shap.plots.waterfall(
            self.shap_values_display[sample_index],
            max_display=CONFIG["max_features_display"],
            show=False
        )
        plt.title(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {output_path}")
        
        plt.close()
    
    def plot_dependence(
        self,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Create a dependence plot showing feature-SHAP relationship.
        
        Parameters:
        -----------
        feature : str or int
            Feature name or index to analyze
        interaction_feature : str or int, optional
            Feature for color-coding interaction effects
        output_path : str, optional
            Path to save the plot
        title : str, optional
            Plot title
        """
        if self.shap_values is None:
            raise ValueError("Call calculate_shap_values() first")
        
        # Get feature index
        if isinstance(feature, str):
            feature_idx = self.feature_names.index(feature)
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = self.feature_names[feature_idx]
        
        title = title or f"SHAP Dependence: {feature_name}"
        
        plt.figure(figsize=(10, 6))
        self.shap.plots.scatter(
            self.shap_values_display[:, feature_idx],
            color=self.shap_values_display if interaction_feature is None else None,
            show=False
        )
        plt.title(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
            logger.info(f"Dependence plot saved to {output_path}")
        
        plt.close()
    
    def create_force_plot_html(
        self,
        sample_index: int = 0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create an interactive force plot and save as HTML.
        
        Parameters:
        -----------
        sample_index : int
            Index of the sample to explain
        output_path : str, optional
            Path to save the HTML file
        
        Returns:
        --------
        str
            Path to the saved HTML file
        """
        if self.shap_values is None:
            raise ValueError("Call calculate_shap_values() first")
        
        sample_shap = self.shap_values_display[sample_index]
        
        force_plot = self.shap.force_plot(
            sample_shap.base_values,
            sample_shap.values,
            sample_shap.data,
            feature_names=self.feature_names
        )
        
        if output_path:
            self.shap.save_html(output_path, force_plot)
            logger.info(f"Force plot HTML saved to {output_path}")
        
        return output_path
    
    def get_top_features(
        self,
        n_features: int = 10
    ) -> pd.DataFrame:
        """
        Get the top N most important features by mean |SHAP|.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature names and importance scores
        """
        if self.shap_values is None:
            raise ValueError("Call calculate_shap_values() first")
        
        mean_abs_shap = np.abs(self.shap_values_display.values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).head(n_features)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df[['rank', 'feature', 'mean_abs_shap']]
    
    def generate_all_explanations(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        output_dir: Optional[str] = None,
        prefix: str = "shap"
    ) -> Dict[str, str]:
        """
        Generate all standard SHAP explanations and save to output directory.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to explain
        output_dir : str, optional
            Directory to save outputs (defaults to CONFIG["output_dir"])
        prefix : str
            Prefix for output file names
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping plot type to file path
        """
        output_dir = Path(output_dir or CONFIG["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        fmt = CONFIG["save_format"]
        
        logger.info(f"Generating all SHAP explanations to {output_dir}")
        
        # Calculate SHAP values
        self.calculate_shap_values(X)
        
        # Generate plots
        output_files["feature_importance"] = str(output_dir / f"{prefix}_feature_importance.{fmt}")
        self.plot_feature_importance(output_path=output_files["feature_importance"])
        
        output_files["summary"] = str(output_dir / f"{prefix}_summary.{fmt}")
        self.plot_summary(output_path=output_files["summary"])
        
        output_files["waterfall"] = str(output_dir / f"{prefix}_waterfall.{fmt}")
        self.plot_waterfall(sample_index=0, output_path=output_files["waterfall"])
        
        # Find most important feature for dependence plot
        top_features = self.get_top_features(1)
        top_feature = top_features.iloc[0]['feature']
        output_files["dependence"] = str(output_dir / f"{prefix}_dependence.{fmt}")
        self.plot_dependence(feature=top_feature, output_path=output_files["dependence"])
        
        # Generate HTML force plot
        if CONFIG["create_html"]:
            output_files["force_plot_html"] = str(output_dir / f"{prefix}_force_plot.html")
            self.create_force_plot_html(sample_index=0, output_path=output_files["force_plot_html"])
        
        # Save top features to CSV
        output_files["top_features_csv"] = str(output_dir / f"{prefix}_top_features.csv")
        top_features = self.get_top_features(CONFIG["max_features_display"])
        top_features.to_csv(output_files["top_features_csv"], index=False)
        logger.info(f"Top features saved to {output_files['top_features_csv']}")
        
        logger.info("All explanations generated successfully!")
        return output_files


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def create_demo_data(
    n_samples: int = 1000,
    n_features: int = 10,
    task_type: str = "classification",
    feature_names: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Create synthetic demo data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    task_type : str
        "classification" or "regression"
    feature_names : List[str], optional
        Custom feature names
    
    Returns:
    --------
    Tuple[pd.DataFrame, np.ndarray, List[str]]
        Features, target, and feature names
    """
    from sklearn.datasets import make_classification, make_regression
    
    if feature_names is None:
        # Create meaningful feature names for healthcare context
        default_names = [
            'age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose',
            'heart_rate', 'smoking_status', 'exercise_freq', 'family_history',
            'medication_count', 'sleep_hours', 'stress_level', 'diet_score',
            'alcohol_consumption', 'previous_conditions'
        ]
        feature_names = default_names[:n_features] if n_features <= len(default_names) else \
                        default_names + [f"feature_{i}" for i in range(len(default_names), n_features)]
    
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(8, n_features),
            n_redundant=min(2, n_features - 8) if n_features > 8 else 0,
            n_classes=2,
            random_state=CONFIG["random_state"]
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(8, n_features),
            noise=10,
            random_state=CONFIG["random_state"]
        )
    
    df = pd.DataFrame(X, columns=feature_names)
    
    return df, y, feature_names


def train_demo_model(
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: str = "classification",
    model_type: str = "tree"
) -> Any:
    """
    Train a demo model for testing.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : np.ndarray
        Target
    task_type : str
        "classification" or "regression"
    model_type : str
        "tree", "linear", or "gradient_boosting"
    
    Returns:
    --------
    Any
        Trained model
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
    )
    
    if task_type == "classification":
        if model_type == "tree":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=CONFIG["random_state"])
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, random_state=CONFIG["random_state"])
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=CONFIG["random_state"])
    else:
        if model_type == "tree":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=CONFIG["random_state"])
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, random_state=CONFIG["random_state"])
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    # Log performance
    if task_type == "classification":
        accuracy = model.score(X_test, y_test)
        logger.info(f"Demo model accuracy: {accuracy:.3f}")
    else:
        r2 = model.score(X_test, y_test)
        logger.info(f"Demo model R² score: {r2:.3f}")
    
    return model, X_train, X_test, y_train, y_test


def run_demo():
    """
    Run a complete demo of SHAP explanations with synthetic data.
    """
    print("=" * 70)
    print("SHAP Explanations Demo")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Create demo data
    logger.info("Creating demo dataset...")
    X, y, feature_names = create_demo_data(
        n_samples=1000,
        n_features=10,
        task_type=CONFIG["task_type"]
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    print()
    
    # Train model
    logger.info("Training demo model...")
    model, X_train, X_test, y_train, y_test = train_demo_model(
        X, y,
        task_type=CONFIG["task_type"],
        model_type=CONFIG["model_type"]
    )
    
    # Create explainer
    logger.info("Creating SHAP explainer...")
    explainer = SHAPExplainer(
        model=model,
        background_data=X_train,
        feature_names=feature_names,
        task_type=CONFIG["task_type"],
        model_type=CONFIG["model_type"]
    )
    
    # Generate all explanations
    output_dir = Path(CONFIG["output_dir"])
    output_files = explainer.generate_all_explanations(
        X=X_test,
        output_dir=output_dir,
        prefix="demo"
    )
    
    # Print summary
    print()
    print("=" * 70)
    print("SHAP Explanations Generated Successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for plot_type, file_path in output_files.items():
        print(f"  - {plot_type}: {Path(file_path).name}")
    
    print("\nTop Features by SHAP Importance:")
    top_features = explainer.get_top_features(5)
    for _, row in top_features.iterrows():
        print(f"  {row['rank']}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    print("=" * 70)
    
    return explainer, output_files


# =============================================================================
# AZURE ML INTEGRATION EXAMPLE
# =============================================================================

def azure_ml_example_code() -> str:
    """
    Return example code for using SHAP with Azure ML models.
    """
    return '''
# =============================================================================
# Example: SHAP with Azure ML AutoML or Custom Models
# =============================================================================

import pandas as pd
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from shap_demo import SHAPExplainer, CONFIG

# -----------------------------------------------------------------------------
# 1. Connect to Azure ML Workspace
# -----------------------------------------------------------------------------

# Option A: Using DefaultAzureCredential (recommended)
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="YOUR_SUBSCRIPTION_ID",
    resource_group_name="YOUR_RESOURCE_GROUP",
    workspace_name="YOUR_WORKSPACE_NAME"
)

# Option B: Using environment variables
# Set these environment variables: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME
# ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# -----------------------------------------------------------------------------
# 2. Download and Load Your Model
# -----------------------------------------------------------------------------

# Download registered model
model_name = "your-model-name"
model_version = "1"

model_download_path = ml_client.models.download(
    name=model_name,
    version=model_version,
    download_path="./downloaded_model"
)

# Load the model using MLflow
model = mlflow.sklearn.load_model(model_download_path)
# OR for pyfunc models:
# model = mlflow.pyfunc.load_model(model_download_path)

# -----------------------------------------------------------------------------
# 3. Load Your Data
# -----------------------------------------------------------------------------

# Option A: From local file
data = pd.read_csv("your_data.csv")
target_column = "your_target_column"

X = data.drop(columns=[target_column])
y = data[target_column]

# Option B: From Azure ML Data Asset
# data_asset = ml_client.data.get(name="your-data-asset", version="1")
# data = pd.read_csv(data_asset.path)

# -----------------------------------------------------------------------------
# 4. Generate SHAP Explanations
# -----------------------------------------------------------------------------

# Update configuration for your use case
CONFIG.update({
    "target_column": target_column,
    "task_type": "classification",  # or "regression"
    "model_type": "tree",           # "tree", "linear", "kernel"
    "output_dir": "./explanations_azure",
    "sample_size": 100,
})

# Create explainer
explainer = SHAPExplainer(
    model=model,
    background_data=X.sample(min(500, len(X))),  # Sample for efficiency
    feature_names=list(X.columns),
    task_type=CONFIG["task_type"],
    model_type=CONFIG["model_type"]
)

# Generate all explanations
output_files = explainer.generate_all_explanations(
    X=X,
    output_dir=CONFIG["output_dir"],
    prefix="azure_model"
)

# Get top features for reporting
top_features = explainer.get_top_features(10)
print("Top 10 Most Important Features:")
print(top_features.to_string(index=False))

# -----------------------------------------------------------------------------
# 5. Optional: Upload Explanations to Azure ML
# -----------------------------------------------------------------------------

# from azure.ai.ml.entities import Data
# from azure.ai.ml.constants import AssetTypes
# 
# explanation_data = Data(
#     name="model-explanations",
#     path=CONFIG["output_dir"],
#     type=AssetTypes.URI_FOLDER,
#     description="SHAP explanations for model"
# )
# ml_client.data.create_or_update(explanation_data)

print("SHAP explanations generated successfully!")
'''


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data
  python shap_demo.py

  # Use custom data and model
  python shap_demo.py --data-path data.csv --model-path model.pkl --target target_column

  # Specify task and model type
  python shap_demo.py --task-type regression --model-type gradient_boosting

  # Custom output directory
  python shap_demo.py --output-dir ./my_explanations --prefix my_model
        """
    )
    
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model (pickle or MLflow format)"
    )
    parser.add_argument(
        "--target", type=str, default=CONFIG["target_column"],
        help=f"Target column name (default: {CONFIG['target_column']})"
    )
    parser.add_argument(
        "--task-type", type=str, default=CONFIG["task_type"],
        choices=["classification", "regression"],
        help=f"Task type (default: {CONFIG['task_type']})"
    )
    parser.add_argument(
        "--model-type", type=str, default=CONFIG["model_type"],
        choices=["tree", "linear", "kernel", "deep"],
        help=f"Model type for SHAP explainer (default: {CONFIG['model_type']})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=CONFIG["output_dir"],
        help=f"Output directory (default: {CONFIG['output_dir']})"
    )
    parser.add_argument(
        "--prefix", type=str, default="shap",
        help="Prefix for output file names (default: shap)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=CONFIG["sample_size"],
        help=f"Number of samples for SHAP analysis (default: {CONFIG['sample_size']})"
    )
    parser.add_argument(
        "--show-azure-example", action="store_true",
        help="Show example code for Azure ML integration"
    )
    parser.add_argument(
        "--log-level", type=str, default=CONFIG["log_level"],
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging level (default: {CONFIG['log_level']})"
    )
    
    return parser.parse_args()


def load_user_data(data_path: str, target: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load user's data from file."""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    
    X = df.drop(columns=[target])
    y = df[target].values
    feature_names = list(X.columns)
    
    logger.info(f"Loaded {len(df)} rows with {len(feature_names)} features")
    
    return X, y, feature_names


def load_user_model(model_path: str):
    """Load user's model from file."""
    import pickle
    
    model_path = Path(model_path)
    
    if model_path.is_dir():
        # Assume MLflow model
        logger.info(f"Loading MLflow model from {model_path}")
        import mlflow
        return mlflow.sklearn.load_model(str(model_path))
    
    elif model_path.suffix in ['.pkl', '.pickle']:
        logger.info(f"Loading pickle model from {model_path}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    elif model_path.suffix == '.joblib':
        logger.info(f"Loading joblib model from {model_path}")
        import joblib
        return joblib.load(model_path)
    
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Update configuration from arguments
    CONFIG.update({
        "target_column": args.target,
        "task_type": args.task_type,
        "model_type": args.model_type,
        "output_dir": args.output_dir,
        "sample_size": args.sample_size,
        "log_level": args.log_level,
    })
    
    # Reconfigure logging
    global logger
    logger = setup_logging(CONFIG["log_level"])
    
    # Show Azure example if requested
    if args.show_azure_example:
        print(azure_ml_example_code())
        return 0
    
    try:
        if args.data_path and args.model_path:
            # Use user's data and model
            print("=" * 70)
            print("SHAP Explanations for Custom Model")
            print("=" * 70)
            print(f"Data: {args.data_path}")
            print(f"Model: {args.model_path}")
            print()
            
            # Load data
            X, y, feature_names = load_user_data(args.data_path, args.target)
            
            # Load model
            model = load_user_model(args.model_path)
            
            # Split for background data
            from sklearn.model_selection import train_test_split
            X_train, X_test, _, _ = train_test_split(
                X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
            )
            
            # Create explainer
            explainer = SHAPExplainer(
                model=model,
                background_data=X_train,
                feature_names=feature_names,
                task_type=CONFIG["task_type"],
                model_type=CONFIG["model_type"]
            )
            
            # Generate explanations
            output_files = explainer.generate_all_explanations(
                X=X_test,
                output_dir=CONFIG["output_dir"],
                prefix=args.prefix
            )
            
            print("\n" + "=" * 70)
            print("SHAP Explanations Generated Successfully!")
            print("=" * 70)
            
        else:
            # Run demo
            run_demo()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())