@echo off
REM ============================================================
REM Submit RAI Dashboard Job to Azure ML Studio
REM ============================================================
REM 
REM This script submits a Responsible AI dashboard job for the
REM Secondary CVD Risk model to Azure ML Studio.
REM
REM Prerequisites:
REM   1. Azure CLI installed and logged in (az login)
REM   2. Python environment with Azure ML SDK v2 installed
REM   3. Configure rai_config.yaml with your workspace details
REM      OR set environment variables:
REM        - AZURE_SUBSCRIPTION_ID
REM        - AZURE_RESOURCE_GROUP
REM        - AZURE_ML_WORKSPACE
REM
REM Usage:
REM   submit_rai_job.bat           - Submit job, don't wait
REM   submit_rai_job.bat --wait    - Submit and wait for completion
REM
REM ============================================================

echo.
echo ============================================================
echo  Azure ML - Responsible AI Dashboard Job Submission
echo ============================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and add it to your PATH
    exit /b 1
)

REM Check if Azure CLI is logged in
echo Checking Azure CLI authentication...
az account show >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Azure CLI not logged in. Running 'az login'...
    az login
    if errorlevel 1 (
        echo ERROR: Azure CLI login failed
        exit /b 1
    )
)

echo Azure CLI authenticated successfully
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if config file exists
if not exist "rai_config.yaml" (
    echo ERROR: rai_config.yaml not found
    echo Please create rai_config.yaml with your Azure ML workspace settings
    echo See rai_config.yaml.sample for an example
    exit /b 1
)

echo Configuration file found: rai_config.yaml
echo.

REM Check for required packages
echo Checking Python dependencies...
python -c "from azure.ai.ml import MLClient" 2>nul
if errorlevel 1 (
    echo.
    echo Installing required packages...
    pip install azure-ai-ml azure-identity python-dotenv pyyaml
)

REM Run the submission script
echo.
echo Submitting RAI Dashboard job...
echo.

if "%1"=="--wait" (
    python submit_rai_job.py --config rai_config.yaml --wait
) else if "%1"=="-w" (
    python submit_rai_job.py --config rai_config.yaml --wait
) else (
    python submit_rai_job.py --config rai_config.yaml %*
)

if errorlevel 1 (
    echo.
    echo ERROR: Job submission failed
    exit /b 1
)

echo.
echo ============================================================
echo  Job Submission Complete
echo ============================================================
echo.
echo To view the RAI dashboard:
echo   1. Go to Azure ML Studio
echo   2. Navigate to Jobs
echo   3. Find the RAI_insights_* experiment
echo   4. Click on the completed job
echo   5. Select "Responsible AI" tab
echo.
