<#
.SYNOPSIS
    Submit RAI Dashboard Job to Azure ML Studio

.DESCRIPTION
    This script submits a Responsible AI dashboard job for the
    Secondary CVD Risk model to Azure ML Studio.

.PARAMETER Wait
    If specified, wait for job completion

.PARAMETER Config
    Path to configuration file (default: rai_config.yaml)

.EXAMPLE
    .\submit_rai_job.ps1
    Submit job without waiting

.EXAMPLE
    .\submit_rai_job.ps1 -Wait
    Submit job and wait for completion

.NOTES
    Prerequisites:
    - Azure CLI installed and logged in (az login)
    - Python environment with Azure ML SDK v2
    - rai_config.yaml configured OR environment variables set:
        AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
#>

param(
    [switch]$Wait,
    [string]$Config = "rai_config.yaml"
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Azure ML - Responsible AI Dashboard Job Submission" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check Azure CLI authentication
Write-Host "Checking Azure CLI authentication..." -ForegroundColor Yellow
try {
    $account = az account show 2>&1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) {
        throw "Not logged in"
    }
    Write-Host "  Logged in as: $($account.user.name)" -ForegroundColor Green
    Write-Host "  Subscription: $($account.name)" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "WARNING: Azure CLI not logged in. Running 'az login'..." -ForegroundColor Yellow
    az login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Azure CLI login failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Check config file
if (-not (Test-Path $Config)) {
    Write-Host "ERROR: Configuration file not found: $Config" -ForegroundColor Red
    Write-Host "Please create $Config with your Azure ML workspace settings" -ForegroundColor Yellow
    exit 1
}

Write-Host "Configuration file: $Config" -ForegroundColor Green
Write-Host ""

# Check Python dependencies
Write-Host "Checking Python dependencies..." -ForegroundColor Yellow
$checkResult = python -c "from azure.ai.ml import MLClient" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing required packages..." -ForegroundColor Yellow
    pip install azure-ai-ml azure-identity python-dotenv pyyaml
}
Write-Host "  Dependencies OK" -ForegroundColor Green

Write-Host ""
Write-Host "Submitting RAI Dashboard job..." -ForegroundColor Yellow
Write-Host ""

# Build arguments
$args = @("submit_rai_job.py", "--config", $Config)
if ($Wait) {
    $args += "--wait"
}

# Run the submission script
python @args

$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Job submission failed" -ForegroundColor Red
    exit $exitCode
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Job Submission Complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To view the RAI dashboard:" -ForegroundColor Cyan
Write-Host "  1. Go to Azure ML Studio" -ForegroundColor White
Write-Host "  2. Navigate to Jobs" -ForegroundColor White
Write-Host "  3. Find the RAI_insights_* experiment" -ForegroundColor White
Write-Host "  4. Click on the completed job" -ForegroundColor White
Write-Host "  5. Select 'Responsible AI' tab" -ForegroundColor White
Write-Host ""
