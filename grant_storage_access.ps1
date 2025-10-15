# Quick script to grant yourself Storage Blob Data Contributor role
# This allows local development to work with managed identity-enabled storage

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Grant Storage Access to Your User Account" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Load .env
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

$subscriptionId = $env:AZURE_SUBSCRIPTION_ID
$resourceGroup = $env:AZURE_RESOURCE_GROUP
$workspace = $env:AZURE_ML_WORKSPACE

Write-Host "Getting workspace storage account..." -ForegroundColor Yellow

# Get workspace info
$wsInfo = az ml workspace show --name $workspace --resource-group $resourceGroup 2>&1 | ConvertFrom-Json

if (-not $wsInfo) {
    Write-Host "Failed to get workspace info" -ForegroundColor Red
    exit 1
}

$storageAccountId = $wsInfo.storage_account
$storageAccountName = $storageAccountId -replace '.*/([^/]+)$', '$1'

Write-Host "Storage Account: $storageAccountName" -ForegroundColor Green
Write-Host ""

# Get current user
$currentUser = az ad signed-in-user show 2>&1 | ConvertFrom-Json
$userPrincipalName = $currentUser.userPrincipalName

Write-Host "Your User: $userPrincipalName" -ForegroundColor Green
Write-Host ""

Write-Host "Granting 'Storage Blob Data Contributor' role..." -ForegroundColor Yellow
Write-Host "This command will run:" -ForegroundColor Gray
Write-Host ""
$command = "az role assignment create --role 'Storage Blob Data Contributor' --assignee $userPrincipalName --scope $storageAccountId"
Write-Host $command -ForegroundColor White
Write-Host ""

$response = Read-Host "Proceed? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y') {
    az role assignment create `
        --role "Storage Blob Data Contributor" `
        --assignee $userPrincipalName `
        --scope $storageAccountId
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Role assignment successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Wait 5-10 minutes for permissions to propagate, then run:" -ForegroundColor Yellow
        Write-Host "  .\run_training.bat" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "✗ Role assignment failed" -ForegroundColor Red
        Write-Host "You may need to ask your Azure administrator for help" -ForegroundColor Yellow
    }
} else {
    Write-Host "Cancelled" -ForegroundColor Yellow
}

Write-Host ""
