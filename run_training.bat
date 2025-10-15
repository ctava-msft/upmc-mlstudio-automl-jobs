@echo off
REM Batch script to activate conda environment and run training
echo ======================================================================
echo Azure ML AutoML Training Launcher
echo ======================================================================
echo.

REM Check if .env file exists
if not exist .env (
    echo ERROR: .env file not found!
    echo.
    echo Please create .env file with your Azure credentials:
    echo   AZURE_SUBSCRIPTION_ID=your-subscription-id
    echo   AZURE_RESOURCE_GROUP=your-resource-group
    echo   AZURE_ML_WORKSPACE=your-workspace-name
    echo.
    pause
    exit /b 1
)

REM Check if config.yaml exists
if not exist config.yaml (
    echo ERROR: config.yaml file not found!
    echo.
    echo Please create config.yaml from one of the example files:
    echo   copy config.yaml.sample config.yaml
    echo   OR
    echo   copy config.example.classification.yaml config.yaml
    echo.
    pause
    exit /b 1
)

echo Activating conda environment 'automl_env'...

REM Initialize conda for batch scripts (required for conda activate to work in .bat)
call conda activate automl_env 2>nul

REM Verify activation by checking Python path
python -c "import sys; print('Python path:', sys.executable)" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate conda environment 'automl_env' or Python not available
    echo.
    echo Troubleshooting:
    echo   1. Check if environment exists: conda env list
    echo   2. Create the environment if needed: conda env create -f conda_env_v_1_0_0.yml
    echo   3. Make sure conda is initialized: conda init cmd.exe
    echo.
    pause
    exit /b 1
)

echo ✓ Environment activated

echo.
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not available
    pause
    exit /b 1
)

echo.
echo Checking Azure ML SDK...
python -c "import azureml.core; print('Azure ML SDK version:', azureml.core.VERSION)"

if errorlevel 1 (
    echo.
    echo ERROR: Azure ML SDK not found in environment
    echo.
    echo Please ensure the conda environment has Azure ML SDK installed:
    echo   conda activate automl_env
    echo   pip install azureml-core
    echo.
    pause
    exit /b 1
)

echo ✓ Azure ML SDK available

echo.
echo ======================================================================
echo Checking Azure Authentication
echo ======================================================================
echo.

REM Check Azure authentication by trying to get account name
echo Checking Azure CLI login status...
for /f "tokens=*" %%a in ('az account show --query name -o tsv 2^>nul') do set ACCOUNT_NAME=%%a

if defined ACCOUNT_NAME (
    echo ✓ Azure CLI authentication: OK
    echo ✓ Active subscription: %ACCOUNT_NAME%
) else (
    echo.
    echo WARNING: Not logged in to Azure CLI
    echo.
    echo You need to authenticate with Azure before running training.
    echo.
    echo Options:
    echo   1. Run setup script: powershell -ExecutionPolicy Bypass -File setup_azure_auth.ps1
    echo   2. Manual login: az login
    echo.
    echo Press any key to run the authentication setup script, or Ctrl+C to cancel...
    pause >nul
    echo.
    echo Running Azure authentication setup...
    echo.
    powershell -ExecutionPolicy Bypass -File setup_azure_auth.ps1
    
    if errorlevel 1 (
        echo.
        echo ERROR: Azure authentication failed
        echo.
        echo Please try manual login:
        echo   az login
        echo   az account set --subscription your-subscription-id
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo Verifying authentication...
    for /f "tokens=*" %%a in ('az account show --query name -o tsv 2^>nul') do set ACCOUNT_NAME_VERIFY=%%a
    if defined ACCOUNT_NAME_VERIFY (
        echo ✓ Authentication verified: %ACCOUNT_NAME_VERIFY%
    ) else (
        echo ERROR: Still not authenticated after setup
        echo.
        echo Please run manually:
        echo   az login
        echo   az account set --subscription your-subscription-id
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ======================================================================
echo Starting Training
echo ======================================================================
echo.
python train.py --config config.yaml

if errorlevel 1 (
    echo.
    echo ======================================================================
    echo Training Failed
    echo ======================================================================
    echo.
    echo Common issues:
    echo   1. Storage authentication - Run: powershell -File setup_azure_auth.ps1
    echo   2. Missing .env file - Copy .env.sample to .env and configure
    echo   3. Invalid config.yaml - Run: python validate_config.py config.yaml
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Training Completed Successfully
echo ======================================================================
echo.
pause
