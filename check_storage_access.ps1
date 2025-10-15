# Storage Access Diagnostic Script
# Checks if you have proper storage access for Azure ML workspace

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Azure ML Storage Access Diagnostic" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
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
    Write-Host "✓ Loaded .env file" -ForegroundColor Green
} else {
    Write-Host "✗ .env file not found" -ForegroundColor Red
    exit 1
}

$subscriptionId = $env:AZURE_SUBSCRIPTION_ID
$resourceGroup = $env:AZURE_RESOURCE_GROUP
$workspace = $env:AZURE_ML_WORKSPACE

Write-Host ""
Write-Host "Workspace Configuration:" -ForegroundColor Yellow
Write-Host "  Subscription: $subscriptionId"
Write-Host "  Resource Group: $resourceGroup"
Write-Host "  Workspace: $workspace"
Write-Host ""

# Check if Azure CLI is installed
$azInstalled = Get-Command az -ErrorAction SilentlyContinue
if (-not $azInstalled) {
    Write-Host "✗ Azure CLI is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install from: https://aka.ms/installazurecliwindows" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Azure CLI installed" -ForegroundColor Green
Write-Host ""

# Check if logged in
Write-Host "Checking Azure CLI authentication..." -ForegroundColor Yellow
$account = az account show 2>&1 | ConvertFrom-Json -ErrorAction SilentlyContinue

if (-not $account) {
    Write-Host "✗ Not logged in to Azure CLI" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run: az login" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Logged in as: $($account.user.name)" -ForegroundColor Green
Write-Host "  Current subscription: $($account.name)" -ForegroundColor Gray
Write-Host ""

# Set correct subscription
if ($account.id -ne $subscriptionId) {
    Write-Host "Setting subscription to: $subscriptionId" -ForegroundColor Yellow
    az account set --subscription $subscriptionId
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to set subscription" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Subscription set" -ForegroundColor Green
}

# Get workspace storage account
Write-Host ""
Write-Host "Retrieving workspace storage account..." -ForegroundColor Yellow

$workspaceInfo = az ml workspace show --name $workspace --resource-group $resourceGroup 2>&1 | ConvertFrom-Json -ErrorAction SilentlyContinue

if (-not $workspaceInfo) {
    Write-Host "✗ Failed to get workspace information" -ForegroundColor Red
    Write-Host "  Make sure the workspace name and resource group are correct" -ForegroundColor Gray
    exit 1
}

# Extract storage account name from the full resource ID
$storageAccountId = $workspaceInfo.storage_account
$storageAccountName = $storageAccountId -replace '.*/([^/]+)$', '$1'

Write-Host "✓ Storage Account: $storageAccountName" -ForegroundColor Green
Write-Host ""

# Test storage access with managed identity (user auth)
Write-Host "Testing storage access with your user identity..." -ForegroundColor Yellow
Write-Host "  This tests if you have 'Storage Blob Data Contributor' role" -ForegroundColor Gray
Write-Host ""

$testResult = az storage blob list `
    --account-name $storageAccountName `
    --container-name azureml `
    --auth-mode login `
    --max-results 1 `
    2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ SUCCESS! You have proper storage access" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run: .\run_training.bat" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "✗ FAILED! You do not have proper storage access" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error details:" -ForegroundColor Yellow
    Write-Host $testResult -ForegroundColor Gray
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Yellow
    Write-Host "SOLUTION: Grant Storage Blob Data Contributor Role" -ForegroundColor Yellow
    Write-Host "=" * 70 -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Using Azure Portal (Recommended)" -ForegroundColor Cyan
    Write-Host "  1. Go to: https://portal.azure.com" 
    Write-Host "  2. Navigate to storage account: $storageAccountName"
    Write-Host "  3. Click 'Access Control (IAM)' in the left menu"
    Write-Host "  4. Click '+ Add' → 'Add role assignment'"
    Write-Host "  5. Select role: 'Storage Blob Data Contributor'"
    Write-Host "  6. Click 'Next'"
    Write-Host "  7. Click '+ Select members'"
    Write-Host "  8. Search for and select your user: $($account.user.name)"
    Write-Host "  9. Click 'Select' → 'Review + assign' → 'Review + assign'"
    Write-Host "  10. Wait 5-10 minutes for permissions to propagate"
    Write-Host ""
    Write-Host "Option 2: Using Azure CLI" -ForegroundColor Cyan
    Write-Host "  Run this command (replace USER_EMAIL with your email):" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  az role assignment create ``" -ForegroundColor White
    Write-Host "    --role 'Storage Blob Data Contributor' ``" -ForegroundColor White
    Write-Host "    --assignee USER_EMAIL ``" -ForegroundColor White
    Write-Host "    --scope /subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.Storage/storageAccounts/$storageAccountName" -ForegroundColor White
    Write-Host ""
    Write-Host "After granting access, wait 5-10 minutes, then run this script again to verify." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "=" * 70 -ForegroundColor Cyan
