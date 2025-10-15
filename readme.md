# Azure Machine Learning AutoML Training Framework

A **generic, configurable framework** for training machine learning models using Azure Machine Learning AutoML. This framework supports any dataset, task type (regression, classification, forecasting), and provides extensive feature engineering capabilities through simple YAML configuration files.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Feature Engineering](#feature-engineering)
- [Examples](#examples)
- [Usage Workflow](#usage-workflow)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Features

- ✅ **Model Agnostic**: Train any type of model (regression, classification, forecasting)
- ✅ **Configurable Feature Engineering**: Binning, scaling, encoding, datetime features, interactions, and custom transformations
- ✅ **YAML-Based Configuration**: No code changes needed - configure everything via config files
- ✅ **Multiple Data Formats**: Supports CSV, Parquet, and Excel files
- ✅ **Data Validation**: Automatic handling of missing values and duplicates
- ✅ **Flexible Deployment**: Deploy models to managed endpoints
- ✅ **Comprehensive Logging**: Track all operations with configurable logging
- ✅ **Configuration Validation**: Built-in validator to catch errors before training

## Prerequisites

1. **Azure Machine Learning workspace** - Active workspace in your subscription
2. **Azure compute cluster** - Configured compute cluster for training
3. **Docker Desktop** (optional) - For local deployment testing
4. **Python 3.8 or higher** - Installed on your system

## Setup Environment

### Using Conda Environment

1. Create and activate a conda environment from the provided environment file:
```bash
conda env create -f conda_env_v_1_0_0.yml
conda activate automl_env
```

**Note**: The conda environment uses Python 3.9 and includes Azure ML SDK v1 (`azureml-train-automl-runtime==1.60.0`), which is required for the current `train.py` script. 

### Configure Azure Credentials

3. Configure Azure credentials by copying `.env.sample` to `.env` and updating with your values:
```bash
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_ML_WORKSPACE=your-workspace-name
```

## Quick Start

> 💡 **Windows Users**: After completing setup steps 1-4, simply run `cmd /c run_training.bat` (PowerShell) or `run_training.bat` (Command Prompt) to start training! This batch script automatically handles environment activation, authentication checks, and training execution.

### 1. Configure Your Training

Use `config.yaml.sample` file or use one of the provided examples:

```bash
# Copy default
cp config.yaml.sample config.yaml

# OR Copy an example configuration e.g. classification
cp config.example.classification.yaml config.yaml
```

Edit the configuration file to specify:
- Your data path and target column
- Task type (regression/classification/forecasting)
- Feature engineering steps
- AutoML settings
- Output preferences

### 2. Validate Configuration (Optional but Recommended)

Before training, validate your configuration file:

```bash
python validate_config.py config.yaml
```

This will check for common errors and provide helpful warnings.

### 3. Configure Azure Storage Authentication

**IMPORTANT**: Azure ML SDK v1 requires storage authentication for ALL dataset operations (loading existing datasets OR uploading new ones).

**For Workspaces Using Managed Identity Storage:**
If your workspace storage is configured with managed identity (Microsoft Entra ID), you must:
1. Have the **Storage Blob Data Contributor** role assigned to YOUR user account on the workspace's storage account
2. Authenticate using Azure CLI or interactive browser login

**Quick Setup** (Run this script):
```powershell
.\setup_azure_auth.ps1
```

This script will:
- Login to Azure
- Test storage access
- Provide instructions to assign Storage Blob Data Contributor role if needed

**Manual Setup**:
```powershell
# 1. Login to Azure CLI (required for managed identity storage)
az login --tenant tenant_id

# 2. Set your subscription
az account set --subscription <your-subscription-id>

# 3. Grant yourself Storage Blob Data Contributor role (if you don't have it):
#    a. Go to Azure Portal → Storage Account (for your ML workspace)
#    b. Click "Access Control (IAM)"
#    c. Click "+ Add" → "Add role assignment"
#    d. Select role: "Storage Blob Data Contributor"
#    e. Assign to: Your user account
#    f. Save and wait 5-10 minutes for propagation
```

**Verify Storage Access:**
```powershell
# Get your workspace storage account name from Azure Portal, then test:
az storage blob list --account-name <storage-account-name> --container-name azureml --auth-mode login
```

### 4. Run Training

**✅ Recommended: Use the Batch Script (Windows)**

The easiest way to run training is using the provided batch script:

```powershell
# From PowerShell (recommended)
cmd /c run_training.bat

# Or from Command Prompt
run_training.bat
```

**What this script does automatically:**
- ✅ Validates `.env` and `config.yaml` files exist
- ✅ Activates the conda environment (`automl_env`)
- ✅ Verifies Azure ML SDK installation
- ✅ **Checks Azure CLI authentication** (runs `setup_azure_auth.ps1` if needed)
- ✅ Executes the training with your `config.yaml`
- ✅ Provides helpful error messages if something goes wrong

**Alternative: Manual Execution**

If you prefer manual control or are on Linux/Mac:

1. **Ensure Azure authentication first** (critical for storage access):
   ```powershell
   # Windows PowerShell
   .\setup_azure_auth.ps1
   
   # Linux/Mac or manual
   az login
   az account set --subscription <your-subscription-id>
   ```

2. **Run training**:
   ```bash
   python train.py --config config.yaml
   ```

**What the Training Script Does:**
1. Load configuration from the YAML file
2. Connect to your Azure ML workspace using `.env` credentials
3. Load and validate your data
4. Apply configured feature engineering transformations
5. Upload and register the dataset in Azure ML (requires storage authentication)
6. Configure and submit the AutoML experiment
7. Monitor training progress
8. Register the best model (if configured)

> ⚠️ **Important**: Azure storage authentication is **required** for dataset operations. The `run_training.bat` script handles this automatically, but manual execution requires running `az login` first.

## Configuration Guide

All training behavior is controlled via the `config.yaml` file. See the comprehensive template `config.yaml` for all available options.

### Key Configuration Sections

#### 1. Experiment Settings
```yaml
experiment:
  name: "my-experiment"
  description: "My AutoML experiment"
```

#### 2. Data Configuration

**Option 1: Use Existing Registered Dataset (Recommended for Production)**
```yaml
data:
  use_existing_dataset: true
  dataset_name: "my_dataset"  # Name from Azure ML workspace
  dataset_version: 1  # Or "latest" for most recent version
  label_column: "target"  # Column to predict
```

**Option 2: Upload from Local File**
```yaml
data:
  use_existing_dataset: false
  input_path: "./data/my_data.csv"
  dataset_name: "my_dataset"
  label_column: "target"  # Column to predict
  upload_to_datastore: false  # false=simple path, true=timestamped path
  columns_to_drop: ["id", "timestamp"]  # Optional
```

**Note**: 
- **Option 1** is recommended when data is already uploaded to Azure ML (e.g., via Azure ML Studio). No local file or feature engineering is performed.
- **Option 2** requires storage authentication. When `upload_to_datastore` is `false`, files upload to `data/local/`. When `true`, to timestamped paths `data/YYYYMMDD_HHMMSS/`.

#### 3. Feature Engineering

**Binning** - Create categorical bins from numeric columns:
```yaml
feature_engineering:
  binning:
    enabled: true
    features:
      - column: "age"
        bins: [0, 18, 35, 50, 65, 100]
        labels: ["Child", "Young", "Adult", "Middle", "Senior"]
        new_column: "age_group"
```

**Datetime Features** - Extract components from date columns:
```yaml
feature_engineering:
  datetime_features:
    enabled: true
    columns:
      - column: "order_date"
        extract: ["year", "month", "day", "dayofweek"]
```

**Interactions** - Create interaction features:
```yaml
feature_engineering:
  interactions:
    enabled: true
    feature_pairs:
      - ["feature1", "feature2"]
```

**Custom Transformations** - Apply Python expressions:
```yaml
feature_engineering:
  custom:
    enabled: true
    transformations:
      - name: "bmi"
        expression: "df['weight'] / (df['height'] ** 2)"
```

#### 4. AutoML Configuration
```yaml
automl:
  task: "regression"  # or "classification", "forecasting"
  primary_metric: "rmse"
  training:
    experiment_timeout_minutes: 60
    n_cross_validations: 5
```

### Monitoring

After submission, monitor the experiment:
- Azure ML Studio URL will be printed in the console
- View real-time metrics, logs, and models in Azure ML Studio
- Training logs are saved to `./logs/training.log` (if configured)

## Examples

The repository includes several example configurations to get you started:

### 1. Classification (`config.example.classification.yaml`)
Simple binary or multi-class classification setup with minimal configuration.

**Use for**: Customer churn, fraud detection, image classification, etc.

```bash
cp config.example.classification.yaml config.yaml
# Edit my-classification.yaml
python train.py --config config.yaml
```

### 2. Regression (`config.example.regression.yaml`)
Generic regression template for predicting continuous values.

**Use for**: Price prediction, demand forecasting, risk scoring, etc.

```bash
cp config.example.regression.yaml config.yaml
# Edit my-regression.yaml
python train.py --config config.yaml
```

### 3. Time Series (`config.example.timeseries.yaml`)
Forecasting template for time-based predictions.

**Use for**: Sales forecasting, stock prices, sensor data, etc.

```bash
cp config.example.timeseries.yaml config.yaml
# Edit my-timeseries.yaml
python train.py --config config.yaml
```

### 4. Surgical Cases (`config.example.surgical-cases.yaml`)
Specific example showing surgical case time prediction with custom binning.

**Demonstrates**: Feature engineering with binning for domain-specific use case.

### 5. Secondary CVD Risk (`config.example.secondary-cvd.yaml`)
Complex example with multiple feature engineering techniques.

**Demonstrates**: 
- Age and BMI binning
- Datetime feature extraction
- Feature interactions
- Custom transformations

## Usage Workflow

### Step 1: Prepare Your Data

Supported formats: CSV, Parquet, Excel

```bash
# Place your data in the data directory
mkdir -p data/my_project
cp /path/to/your/data.csv data/my_project/
```

**Data Requirements**:
- Tabular format (rows and columns)
- Must have a target/label column
- Column names without special characters (use underscores)
- Clean or document missing values

### Step 2: Create Configuration

Choose a starting point:

```bash
# For classification
cp config.example.classification.yaml config.yaml

# For regression
cp config.example.regression.yaml config.yaml

# For time series
cp config.example.timeseries.yaml config.yaml

# Or start from scratch
cp config.yaml.sample config.yaml
```

Edit your configuration:
1. Set `data.input_path` to your data file
2. Set `data.label_column` to your target column
3. Configure `compute.cluster_name` with your Azure ML cluster
4. Adjust `automl.training` settings as needed
5. Enable feature engineering if desired

### Step 3: Validate Configuration

Always validate before training:

```bash
python validate_config.py config.yaml
```

The validator checks:
- YAML syntax
- Required fields
- Data file existence
- Configuration consistency
- Provides helpful warnings

### Step 4: Run Training

```bash
python train.py --config config.yaml
```

**What happens**:
1. Configuration loaded from YAML
2. Connects to Azure ML workspace via `.env`
3. Loads and validates data
4. Applies feature engineering
5. Uploads and registers dataset
6. Configures AutoML experiment
7. Submits to compute cluster
8. Monitors progress
9. Registers best model

**Monitoring**:
- Console shows real-time progress
- Azure ML Studio URL printed (click to view in browser)
- Logs saved to `./logs/training.log`
- Can close terminal and check Azure ML Studio anytime

### Step 5: Review Results

After training completes:

```
==========================================
Training Complete!
Best run ID: AutoML_xxxxx
Best model metrics:
  auc_weighted: 0.85
  mean_absolute_error: 2.34
==========================================
Model registered: my_model (version 1)
```

Access your model in:
- **Azure ML Studio**: View all metrics, charts, feature importance
- **Models Registry**: Download or deploy the registered model
- **Logs**: Review detailed execution logs

## Deploying Models

### Create Endpoint

Start Docker Desktop, then create an endpoint by running:

```bash
python deploy.py
```

This script will:
1. Load environment variables from `.env` file
2. Connect to your Azure ML workspace
3. Create a managed online endpoint
4. Deploy the model to the endpoint

### Test Inference

Test the deployed endpoint:

```bash
python inference.py --host 127.0.0.1 --port 32768 --endpoint score
```

## Project Structure

```
.
├── .env                                      # Azure credentials (not in git)
├── .env.sample                              # Template for credentials
├── config.yaml                              # Main configuration template
├── config.example.*.yaml                    # Example configurations
├── train.py                                 # Generic training script
├── feature_engineering.py                   # Feature engineering module
├── deploy.py                                # Model deployment script
├── inference.py                             # Inference testing script
├── requirements.txt                         # Python dependencies
├── data/                                    # Training data directory
│   └── preprocessed/                        # Preprocessed data outputs
├── models/                                  # Saved models
├── logs/                                    # Training logs
└── README.md                                # This file
```

## Advanced Features

### Data Validation

Automatically check and clean your data:
```yaml
data:
  validation:
    check_missing_values: true
    max_missing_percentage: 50  # Drop columns with >50% missing
    check_duplicates: true
```

### Feature Engineering Options

- **Binning**: Convert continuous variables to categories
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Encoding**: One-hot, label, or target encoding
- **Datetime**: Extract year, month, day, dayofweek, etc.
- **Interactions**: Multiply features together
- **Polynomial**: Create squared, cubed features
- **Custom**: Any Python expression

### Model Selection

Control which models are trained:
```yaml
automl:
  models:
    allowed: ["LightGBM", "XGBoostRegressor"]  # Only these
    blocked: ["ElasticNet"]  # Exclude these
```

### ONNX Export

Enable ONNX-compatible models:
```yaml
automl:
  onnx:
    enable_onnx_compatible_models: true
```

## Output

The best model will be:
- Registered in Azure ML Model Registry (if configured)
- Available for download via Azure ML Studio
- Ready for deployment as a web service
- Logged with all metrics and parameters

## Troubleshooting

### Common Issues

**Issue**: "Missing Azure credentials in .env file"
- **Solution**: Ensure `.env` contains `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, and `AZURE_ML_WORKSPACE`

**Issue**: "Column not found"
- **Solution**: Check that `label_column` and feature engineering columns match your actual data

**Issue**: "Compute cluster not found"
- **Solution**: Verify `compute.cluster_name` in config matches your Azure ML compute cluster name

**Issue**: "File not found"
- **Solution**: Use absolute paths or ensure paths are relative to where you run `train.py`

### Debugging

- Set `log_level: "DEBUG"` in config for verbose output
- Check `./logs/training.log` for detailed logs
- Review experiment details in Azure ML Studio
- Validate your config file with a YAML validator

**Issue**: Conda environment shows wrong Python version (3.12 instead of 3.9)
- **Solution**: Use the batch script provided: `run_training.bat` OR use the full path to Python:
  ```powershell
  # Option 1: Use the helper script
  .\run_training.bat
  
  # Option 2: Call Python directly with full path
  C:\Users\<YourUsername>\AppData\Local\miniforge3\envs\automl_env\python.exe train.py --config config.yaml
  
  # Option 3: Initialize conda for PowerShell (one-time setup)
  conda init powershell
  # Then restart PowerShell and activate normally
  conda activate automl_env
  ```

**Issue**: "ClientAuthenticationError: Server failed to authenticate the request" or "NoAuthenticationInformation"
- **Symptom**: Datastore upload fails with storage authentication errors
- **Solution**: Storage authentication is separate from workspace authentication. Azure ML SDK v1 requires files to be uploaded to a datastore. You need to configure storage credentials for the workspace's default datastore.
  
  **Options to fix**:
  1. Use Azure CLI to login and configure storage access:
     ```powershell
     az login
     az account set --subscription <your-subscription-id>
     ```
  2. Configure storage credentials in Azure Portal for the workspace's storage account
  3. Use managed identity if running from Azure resources
  
  The `upload_to_datastore` flag controls the upload path:
  - `false` (default): Simple path `data/local/` (no timestamp)
  - `true`: Timestamped path `data/YYYYMMDD_HHMMSS/` for versioning

## Common Use Cases and Patterns

### Pattern 1: Simple Prediction Model
Start with minimal configuration, no feature engineering.

```yaml
data:
  input_path: "./data/simple_data.csv"
  label_column: "outcome"

feature_engineering:
  enabled: false

automl:
  task: "classification"
  primary_metric: "auc_weighted"
  training:
    experiment_timeout_minutes: 30
```

### Pattern 2: Feature Engineering for Better Performance

Add domain-specific feature engineering:

```yaml
feature_engineering:
  enabled: true
  binning:
    enabled: true
    features:
      - column: "age"
        bins: [0, 18, 35, 50, 65, 100]
        labels: ["Child", "Young", "Adult", "Middle", "Senior"]
        new_column: "age_group"
  
  custom:
    enabled: true
    transformations:
      - name: "bmi"
        expression: "df['weight'] / ((df['height']/100) ** 2)"
```

### Pattern 3: Time-Based Features

Extract temporal patterns from dates:

```yaml
feature_engineering:
  datetime_features:
    enabled: true
    columns:
      - column: "transaction_date"
        extract: ["year", "month", "dayofweek", "quarter"]
```

### Pattern 4: Feature Interactions

Capture relationships between variables:

```yaml
feature_engineering:
  interactions:
    enabled: true
    feature_pairs:
      - ["temperature", "humidity"]
      - ["price", "quantity"]
```

### Pattern 5: Model Selection and Optimization

Control which models are used:

```yaml
automl:
  models:
    allowed: ["LightGBM", "XGBoostRegressor", "RandomForest"]
    blocked: []
  
  training:
    experiment_timeout_minutes: 120  # More time for better models
    max_concurrent_iterations: 8     # Use more parallelism
```

## Command Reference

### Training Commands
```bash
# Validate configuration
python validate_config.py config.yaml

# Train with specific config
python train.py --config config.yaml

# Train with verbose output
python train.py --config config.yaml 2>&1 | Tee-Object -FilePath training.log
```

### Deployment Commands
```bash
# Deploy model to endpoint
python deploy.py

# Test local endpoint
python inference.py --host 127.0.0.1 --port 32768 --endpoint score
```

### Utility Commands
```bash
# Check Python environment
python --version

# Verify Azure ML SDK installation
python -c "import azureml.core; print(azureml.core.VERSION)"

# List available compute clusters (from Azure CLI)
az ml compute list --resource-group <rg> --workspace-name <ws>
```

## Metrics Reference

### Regression Metrics
- **r2_score**: R² coefficient (0-1, higher is better)
- **normalized_root_mean_squared_error**: Normalized RMSE (0+, lower is better)
- **normalized_mean_absolute_error**: Normalized MAE (0+, lower is better)
- **spearman_correlation**: Spearman's rank correlation (-1 to 1)

### Classification Metrics
- **AUC_weighted**: Area Under ROC Curve (0-1, higher is better)
- **accuracy**: Classification accuracy (0-1, higher is better)
- **precision_score_weighted**: Weighted precision (0-1, higher is better)
- **recall_score_weighted**: Weighted recall (0-1, higher is better)
- **f1_score_weighted**: Weighted F1 score (0-1, higher is better)

### Forecasting Metrics
- **normalized_root_mean_squared_error**: Primary metric for forecasting
- **r2_score**: R² for time series
- **normalized_mean_absolute_error**: MAE for time series

## Tips and Best Practices

### 1. Start Simple, Iterate
- Begin with minimal config and no feature engineering
- Get a baseline model working first
- Add complexity incrementally
- Compare metrics after each change

### 2. Data Quality First
```python
# Before training, inspect your data:
import pandas as pd
df = pd.read_csv('your_data.csv')

# Check shape and types
print(df.shape)
print(df.dtypes)

# Check missing values
print(df.isnull().sum())

# Check target distribution
print(df['target'].describe())
```

### 3. Use Version Control
```bash
# Track your configurations
git add config.my-experiment.yaml
git commit -m "Add config for experiment X"

# Tag successful experiments
git tag v1.0-baseline
```

### 4. Monitor Costs
- Match `max_concurrent_iterations` to cluster node count
- Use shorter `experiment_timeout_minutes` for testing
- Consider cluster auto-shutdown policies
- Review Azure costs regularly

### 5. Feature Engineering Guidelines
- **Domain Knowledge**: Use your understanding of the problem
- **Avoid Leakage**: Don't use future information to predict the past
- **Test Impact**: Compare models with and without each feature
- **Document Choices**: Add comments in YAML explaining why

### 6. Experiment Organization
```bash
# Use descriptive experiment names
experiment:
  name: "churn-model-v2-with-temporal-features"
  description: "Added month and dayofweek features, improved AUC from 0.82 to 0.87"
```

## Contributing

To extend this framework:

### Add New Feature Engineering
1. Edit `feature_engineering.py`
2. Add method to `FeatureEngineer` class
3. Update `config.yaml` template
4. Add example to documentation
5. Test with sample data

### Add New Validation Rules
1. Edit `validate_config.py`
2. Add method to `ConfigValidator` class
3. Call from `validate()` method
4. Add test cases

### Improve Documentation
1. Add examples to this README
2. Update `config.yaml` comments
3. Create new example configs
4. Share learnings and patterns

## Project Files

### Core Framework
- **train.py** - Main training script (no editing needed)
- **feature_engineering.py** - Feature engineering module
- **validate_config.py** - Configuration validation tool
- **deploy.py** - Model deployment script
- **inference.py** - Inference testing script

### Configuration Files
- **config.yaml** - Comprehensive configuration template
- **config.example.classification.yaml** - Classification example
- **config.example.regression.yaml** - Regression example
- **config.example.timeseries.yaml** - Time series forecasting example
- **config.example.surgical-cases.yaml** - Surgical case prediction (specific use case)
- **config.example.secondary-cvd.yaml** - Advanced feature engineering example
- **.env** - Azure credentials (not in git, create from .env.sample)
- **.env.sample** - Template for environment variables

### Documentation
- **README.md** (this file) - Complete documentation

### Directories
- **data/** - Place your training data here
- **data/preprocessed/** - Preprocessed data outputs
- **models/** - Downloaded models
- **logs/** - Training logs

## Quick Command Reference

```bash
# Setup (one time) - Option 1: venv
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Setup (one time) - Option 2: conda (Recommended)
conda env create -f conda_env_v_1_0_0.yml
conda activate automl_env

# Configure credentials
cp .env.sample .env  # Edit with your Azure credentials

# Create config from example
cp config.example.regression.yaml my-config.yaml

# Validate config
python validate_config.py my-config.yaml

# Train model
python train.py --config my-config.yaml

# Deploy model
python deploy.py

# Test endpoint
python inference.py --host 127.0.0.1 --port 32768
```

## Support and Resources

### Documentation Files
- **README.md** (this file) - Complete documentation
- **config.yaml** - Configuration template with all options
- **config.example.*.yaml** - Working examples for different scenarios

### Getting Help

**Local Debugging**:
1. Check `./logs/training.log` for detailed logs
2. Run `python validate_config.py your-config.yaml`
3. Test with small data subset first
4. Use `log_level: "DEBUG"` for verbose output

**Azure ML Studio**:
1. Click the URL printed by training script
2. View experiment runs and metrics
3. Check model registry
4. Review compute cluster status

**Common Issues and Solutions**:
- Configuration errors → Run validator
- Data not found → Check paths are correct
- Compute not ready → Start cluster in Azure
- Poor performance → Try feature engineering
- Out of memory → Reduce data or concurrent iterations

### External Resources
- **Azure ML Documentation**: https://docs.microsoft.com/azure/machine-learning/
- **Azure ML SDK Reference**: https://docs.microsoft.com/python/api/overview/azure/ml/
- **AutoML Overview**: https://docs.microsoft.com/azure/machine-learning/concept-automated-ml
- **Feature Engineering Guide**: https://docs.microsoft.com/azure/machine-learning/how-to-configure-auto-features
- **YAML Syntax**: https://yaml.org/spec/1.2/spec.html

## License

This framework is provided as-is for training Azure ML AutoML models. Refer to your organization's policies for data handling and model deployment.

## Acknowledgments

This framework was developed to provide a generic, reusable solution for training machine learning models with Azure ML AutoML, eliminating the need for custom code for each project.