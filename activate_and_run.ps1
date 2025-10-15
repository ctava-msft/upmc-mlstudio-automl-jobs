# PowerShell script to properly activate conda environment and run training
# This ensures the correct Python and packages are used

# Activate the conda environment
& conda activate automl_env

# Verify Python version
Write-Host "Python version:"
& python --version

# Verify Azure ML SDK
Write-Host "`nAzure ML SDK version:"
& python -c "import azureml.core; print(azureml.core.VERSION)"

# Run the training script
Write-Host "`nStarting training..."
& python train.py --config config.yaml
