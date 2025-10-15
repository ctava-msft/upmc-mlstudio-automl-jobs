# Azure Authentication Setup Script
# This script helps configure Azure authentication for storage access

Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "Azure ML Storage Authentication Setup"
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host ""

# Load environment variables
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$subscriptionId = $env:AZURE_SUBSCRIPTION_ID
$resourceGroup = $env:AZURE_RESOURCE_GROUP
$workspace = $env:AZURE_ML_WORKSPACE

Write-Host "Workspace Configuration:"
Write-Host "  Subscription: $subscriptionId"
Write-Host "  Resource Group: $resourceGroup"
Write-Host "  Workspace: $workspace"
Write-Host ""

# Check if Azure CLI is installed
$azInstalled = Get-Command az -ErrorAction SilentlyContinue
if (-not $azInstalled) {
    Write-Host "ERROR: Azure CLI is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Azure CLI from: https://aka.ms/installazurecliwindows"
    Write-Host ""
    exit 1
}

Write-Host "Azure CLI found: $($azInstalled.Source)" -ForegroundColor Green
Write-Host ""

# Perform Azure login
Write-Host "Step 1: Azure Login" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------"
Write-Host "Logging in to Azure..."
Write-Host ""

az login

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Azure login failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Login successful" -ForegroundColor Green
Write-Host ""

# Set subscription
Write-Host "Step 2: Set Subscription" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------"
Write-Host "Setting active subscription to: $subscriptionId"
Write-Host ""

az account set --subscription $subscriptionId

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to set subscription" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Subscription set" -ForegroundColor Green
Write-Host ""

# Get workspace storage account
Write-Host "Step 3: Get Workspace Storage Account" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------"
Write-Host "Retrieving storage account for workspace..."
Write-Host ""

$workspaceInfo = az ml workspace show --name $workspace --resource-group $resourceGroup --subscription $subscriptionId | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to get workspace information" -ForegroundColor Red
    exit 1
}

$storageAccount = $workspaceInfo.storage_account -replace '.*/([^/]+)$', '$1'
Write-Host "Storage Account: $storageAccount" -ForegroundColor Cyan
Write-Host ""

# Test storage access
Write-Host "Step 4: Test Storage Access" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------"
Write-Host "Testing access to storage account..."
Write-Host ""

$containers = az storage container list --account-name $storageAccount --auth-mode login 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Storage access test failed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You may need additional permissions. To grant Storage Blob Data Contributor role:"
    Write-Host ""
    Write-Host "  1. Go to Azure Portal: https://portal.azure.com"
    Write-Host "  2. Navigate to the storage account: $storageAccount"
    Write-Host "  3. Click 'Access Control (IAM)' in the left menu"
    Write-Host "  4. Click '+ Add' > 'Add role assignment'"
    Write-Host "  5. Select role: 'Storage Blob Data Contributor'"
    Write-Host "  6. Assign to your user account"
    Write-Host "  7. Save and wait a few minutes for propagation"
    Write-Host ""
} else {
    Write-Host "✓ Storage access successful" -ForegroundColor Green
    Write-Host ""
}

Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "Setup Complete"
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host ""
Write-Host "You can now run: .\run_training.bat" -ForegroundColor Cyan
Write-Host ""
