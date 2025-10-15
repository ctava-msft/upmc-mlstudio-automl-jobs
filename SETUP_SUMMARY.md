# Azure ML AutoML Training - Setup Summary

## Current Configuration

### Authentication Setup
- **Method**: Interactive Browser Login (for local development)
- **User-Assigned Managed Identity (UAMI)**: 552118d0-4827-43fb-a5e5-dc4fe16db3d6
  - Has "AI Administrator" role on Azure ML workspace
  - Has "Storage Blob Data Contributor" role on storage account
  - **Note**: UAMI will be used automatically when jobs run on Azure compute

### The Storage Authentication Issue

**Problem**: The workspace storage account uses Managed Identity authentication, but you're running the script locally.

**Solution**: Your user account ALSO needs "Storage Blob Data Contributor" role on the storage account for local development.

## Next Steps

### Option 1: Grant Storage Access to Your User (Recommended for Development)

Run this script to grant yourself storage access:
```powershell
.\grant_storage_access.ps1
```

This will:
1. Identify the workspace's storage account
2. Grant "Storage Blob Data Contributor" role to your user account
3. Allow local script execution to work

**After running, wait 5-10 minutes for permissions to propagate.**

### Option 2: Use the Diagnostic Script

Check your current access level:
```powershell
.\check_storage_access.ps1
```

This will:
- Verify Azure CLI authentication
- Test storage access with your credentials
- Provide detailed instructions if access is missing

### Option 3: Manual Role Assignment

1. Go to Azure Portal: https://portal.azure.com
2. Navigate to the storage account for your ML workspace
3. Click "Access Control (IAM)"
4. Click "+ Add" → "Add role assignment"
5. Select role: "Storage Blob Data Contributor"
6. Select your user account
7. Click "Review + assign"
8. Wait 5-10 minutes for propagation

## Running the Training

Once storage access is configured:

```powershell
# Activate conda environment and run training
.\run_training.bat
```

The script will:
1. ✓ Connect to Azure ML workspace (using your interactive login)
2. ✓ Load and preprocess local data
3. ✓ Upload to datastore (using your storage permissions)
4. ✓ Register dataset
5. ✓ Submit AutoML training job to Azure compute
6. ✓ Training runs on Azure compute (using the UAMI for data access)

## Configuration Files

### config.yaml
- **authentication.method**: "interactive" (for local execution)
- **authentication.managed_identity_client_id**: UAMI client ID (used by Azure compute)
- **data.use_existing_dataset**: false (uploading from local file)
- **data.input_path**: "./data/secondary_cvd_risk_min/secondary-cvd-risk.csv"

### .env
Contains:
- AZURE_SUBSCRIPTION_ID
- AZURE_RESOURCE_GROUP  
- AZURE_ML_WORKSPACE

## Authentication Flow

### Local Execution (Your Machine)
1. Interactive browser login → workspace connection
2. Your user credentials → storage access for upload
3. Dataset registered in Azure ML

### Azure Compute Execution (Training Job)
1. UAMI credentials → workspace access
2. UAMI credentials → storage access for reading data
3. AutoML training proceeds

## Troubleshooting

### "NoAuthenticationInformation" Error
**Cause**: Your user doesn't have storage access
**Solution**: Run `.\grant_storage_access.ps1`

### "PermissionDenied" Error
**Cause**: Storage permissions haven't propagated yet
**Solution**: Wait 5-10 minutes and retry

### Interactive Login Opens Too Many Browser Windows
**Solution**: Close extra windows, the authentication will still work

### Dataset Already Exists Error
**Solution**: The framework will create new versions automatically

## Files Created

- ✅ `run_training.bat` - Main training script runner
- ✅ `check_storage_access.ps1` - Diagnostic tool
- ✅ `grant_storage_access.ps1` - Permission granter
- ✅ `setup_azure_auth.ps1` - Full authentication setup
- ✅ `config.yaml` - Training configuration
- ✅ `train.py` - Main training script (updated with UAMI support)
- ✅ `conda_env_v_1_0_0.yml` - Conda environment specification
- ✅ `readme.md` - Complete documentation

## Quick Start Checklist

- [ ] Azure CLI installed and logged in (`az login`)
- [ ] Conda environment created and activated
- [ ] Storage access granted to your user
- [ ] Wait 5-10 minutes after granting access
- [ ] Run `.\run_training.bat`

## Expected Output

When successful, you'll see:
```
Python 3.9.22
Azure ML SDK version: 1.60.0
Connected to workspace: mlw-25
Data loaded. Shape: (63852, 111)
Feature engineering complete. Final shape: (63852, 123)
File uploaded to datastore path: data/local/preprocessed_data_*.csv
Dataset registered: secondary_cvd_risk_min
AutoML configuration created
Submitting experiment: secondary-cvd-risk
Experiment submitted: <run_id>
```

Then training proceeds on Azure compute using the UAMI.

## Support

For issues:
1. Run `.\check_storage_access.ps1` to diagnose
2. Check Azure Portal for role assignments
3. Verify .env file has correct values
4. Ensure conda environment is activated

---
Last Updated: October 13, 2025
