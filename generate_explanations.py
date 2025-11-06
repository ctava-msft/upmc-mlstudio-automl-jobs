"""
Model Interpretability and Explanation Script
Generates SHAP explanations and feature importance for Azure ML AutoML models
"""

import os
import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from dotenv import load_dotenv
import yaml
import argparse

# Azure ML imports
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, AzureCliCredential

# Interpretability imports
try:
    import shap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from interpret.ext.blackbox import TabularExplainer
    INTERPRETABILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Interpretability libraries not available: {e}")
    print("Install with: pip install shap interpret matplotlib seaborn")
    INTERPRETABILITY_AVAILABLE = False

# Local imports
from feature_engineering import FeatureEngineer, validate_data

# Load environment variables
load_dotenv()


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('./logs/explanations.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_preprocess_data(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Load and preprocess data according to configuration"""
    data_config = config['data']
    
    if data_config.get('use_existing_dataset', False):
        # For existing datasets, you would need to download from Azure ML
        logger.warning("Using existing dataset - you'll need to download the data manually")
        logger.info("Alternatives:")
        logger.info("1. Use the local preprocessed data files in ./data/preprocessed/")
        logger.info("2. Download the dataset from Azure ML Studio")
        logger.info("3. Set use_existing_dataset: false and provide input_path")
        
        # Try to find the most recent preprocessed file
        preprocessed_dir = Path("./data/preprocessed")
        if preprocessed_dir.exists():
            csv_files = list(preprocessed_dir.glob("*.csv"))
            if csv_files:
                latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Using latest preprocessed file: {latest_file}")
                return pd.read_csv(latest_file)
        
        raise FileNotFoundError("No preprocessed data found. Please provide data manually.")
    
    else:
        # Load from local file path
        input_path = data_config['input_path']
        logger.info(f"Loading data from {input_path}")
        
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        elif input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        logger.info(f"Data loaded. Shape: {df.shape}")
        
        # Apply validation
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


def download_best_model(ml_client: MLClient, job_name: str, logger: logging.Logger):
    """Download the best model from a completed AutoML job"""
    try:
        # Get the job details
        job = ml_client.jobs.get(job_name)
        
        if job.status != "Completed":
            raise ValueError(f"Job {job_name} is not completed. Status: {job.status}")
        
        logger.info(f"Job {job_name} completed successfully")
        
        # For AutoML jobs, the best model is typically registered automatically
        # You would need to find the model name from the job outputs
        
        logger.warning("Model download not fully implemented - this is a template")
        logger.info("To download the model:")
        logger.info("1. Go to Azure ML Studio")
        logger.info("2. Navigate to the job results")
        logger.info("3. Download the best model")
        logger.info("4. Load it in this script")
        
        return None
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise


def generate_shap_explanations(model, X, y, feature_names, output_dir: Path, logger: logging.Logger):
    """Generate SHAP explanations for the model"""
    if not INTERPRETABILITY_AVAILABLE:
        logger.error("SHAP not available. Install with: pip install shap")
        return
    
    logger.info("Generating SHAP explanations...")
    
    # Sample data for explanation (SHAP can be slow on large datasets)
    sample_size = min(1000, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    
    try:
        # Create SHAP explainer
        # Note: The explainer type depends on your model type
        # For tree-based models, use TreeExplainer
        # For general models, use Explainer (KernelExplainer)
        
        logger.info("Creating SHAP explainer...")
        explainer = shap.Explainer(model, X_sample)
        
        # Calculate SHAP values for a subset
        explain_size = min(100, len(X_sample))
        X_explain = X_sample.head(explain_size)
        
        logger.info(f"Calculating SHAP values for {explain_size} samples...")
        shap_values = explainer(X_explain)
        
        # Generate summary plots
        logger.info("Creating SHAP visualizations...")
        
        # Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, plot_type='bar', show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Waterfall plot for first prediction
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_values[0], show=False)
        plt.title('SHAP Waterfall Plot (First Sample)')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save SHAP values
        shap_data = {
            'shap_values': shap_values.values,
            'base_values': shap_values.base_values,
            'data': X_explain.values,
            'feature_names': feature_names
        }
        
        with open(output_dir / 'shap_values.pkl', 'wb') as f:
            pickle.dump(shap_data, f)
        
        logger.info("SHAP explanations generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {str(e)}")
        raise


def generate_global_explanations(model, X, feature_names, output_dir: Path, logger: logging.Logger):
    """Generate global explanations using TabularExplainer"""
    if not INTERPRETABILITY_AVAILABLE:
        logger.error("Interpret library not available. Install with: pip install interpret")
        return
    
    try:
        logger.info("Generating global explanations with TabularExplainer...")
        
        # Sample data for explanation
        sample_size = min(1000, len(X))
        X_sample = X.sample(sample_size, random_state=42)
        
        # Create tabular explainer
        explainer = TabularExplainer(model, X_sample, features=feature_names)
        
        # Generate global explanation
        global_explanation = explainer.explain_global()
        
        # Save the explanation
        with open(output_dir / 'global_explanation.pkl', 'wb') as f:
            pickle.dump(global_explanation, f)
        
        # Create feature importance plot
        if hasattr(global_explanation, 'get_ranked_global_values'):
            importance_values = global_explanation.get_ranked_global_values()
            importance_names = global_explanation.get_ranked_global_names()
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_values)), importance_values)
            plt.yticks(range(len(importance_names)), importance_names)
            plt.xlabel('Global Importance')
            plt.title('Global Feature Importance')
            plt.tight_layout()
            plt.savefig(output_dir / 'global_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Global explanations generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating global explanations: {str(e)}")
        raise


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate model explanations for Azure ML AutoML models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--job-name', type=str, help='Azure ML job name to get model from')
    parser.add_argument('--model-path', type=str, help='Local path to saved model')
    parser.add_argument('--output-dir', type=str, default='./explanations', help='Output directory for explanations')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Model Interpretability and Explanation Generation")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load and preprocess data
        df = load_and_preprocess_data(config, logger)
        
        # Prepare features and target
        data_config = config['data']
        label_column = data_config['label_column']
        
        feature_columns = [col for col in df.columns if col != label_column]
        X = df[feature_columns]
        y = df[label_column]
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Features: {len(feature_columns)}")
        logger.info(f"Target column: {label_column}")
        
        # Load model
        model = None
        if args.model_path:
            logger.info(f"Loading model from: {args.model_path}")
            # Add model loading logic here based on your model format
            # model = joblib.load(args.model_path)  # for sklearn models
            # model = pickle.load(open(args.model_path, 'rb'))  # for pickle files
            logger.warning("Model loading not implemented - add your model loading code")
        
        elif args.job_name:
            logger.info(f"Downloading model from job: {args.job_name}")
            
            # Connect to Azure ML
            credential = AzureCliCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
                resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
                workspace_name=os.getenv("AZURE_ML_WORKSPACE")
            )
            
            model = download_best_model(ml_client, args.job_name, logger)
        
        else:
            logger.error("Please provide either --job-name or --model-path")
            return
        
        if model is None:
            logger.warning("No model loaded - generating explanation templates only")
            
            # Create explanation code template
            template_code = f'''
# Model Explanation Template
# Run this code after loading your trained model

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from interpret.ext.blackbox import TabularExplainer

# Load your model
# model = joblib.load('path_to_your_model.pkl')
# or
# model = pickle.load(open('path_to_your_model.pkl', 'rb'))

# Load data (already preprocessed in this script)
X = df[{feature_columns}]
y = df['{label_column}']

# Generate SHAP explanations
explainer = shap.Explainer(model, X.sample(min(1000, len(X))))
shap_values = explainer(X.sample(min(100, len(X))))

# Create plots
shap.summary_plot(shap_values, X.sample(min(100, len(X))), show=False, plot_type='bar')
plt.title('SHAP Feature Importance')
plt.savefig('./explanations/shap_importance.png', dpi=300, bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values, X.sample(min(100, len(X))), show=False)
plt.title('SHAP Summary Plot')
plt.savefig('./explanations/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate global explanations
tabular_explainer = TabularExplainer(model, X.sample(min(1000, len(X))), features={feature_columns})
global_explanation = tabular_explainer.explain_global()

print("Explanations generated successfully!")
'''
            
            with open(output_dir / 'explanation_template.py', 'w') as f:
                f.write(template_code)
            
            logger.info(f"Explanation template saved to: {output_dir / 'explanation_template.py'}")
            
        else:
            # Generate actual explanations
            generate_shap_explanations(model, X, y, feature_columns, output_dir, logger)
            generate_global_explanations(model, X, feature_columns, output_dir, logger)
        
        # Save data summary
        data_summary = {
            'dataset_shape': df.shape,
            'feature_count': len(feature_columns),
            'feature_names': feature_columns,
            'target_column': label_column,
            'target_distribution': y.value_counts().to_dict() if y.dtype == 'object' else None
        }
        
        with open(output_dir / 'data_summary.pkl', 'wb') as f:
            pickle.dump(data_summary, f)
        
        logger.info("=" * 70)
        logger.info("Model explanation generation completed!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()