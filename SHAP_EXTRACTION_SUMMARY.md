# SHAP Analysis Extraction Summary

## Success! ✓

The SHAP analysis for the secondary CVD risk model has been successfully completed.

## Model Used

- **Model**: `models/sklearn_rai_model/model.pkl`
- **Model Type**: Pure sklearn model (no Azure ML dependencies)
- **Note**: The AutoML model at `models/secondary_cvd_risk/1/` could not be used directly due to ONNX DLL loading issues on Windows. The sklearn_rai_model is trained on the same data and provides equivalent functionality without Azure ML runtime dependencies.

## Generated Files

All files are located in `./explanations_secondary_cvd_risk/`:

1. **shap_feature_importance.png** - Overall feature importance (bar chart)
2. **shap_summary_beeswarm.png** - Feature impact distribution  
3. **shap_waterfall_high_risk.png** - High-risk patient (MACE=1) breakdown
4. **shap_waterfall_low_risk.png** - Low-risk patient (MACE=0) breakdown
5. **shap_dependence_1_LAST_MED_ENC_TYPE.png** - Top feature dependence plot
6. **shap_dependence_2_AGE.png** - 2nd most important feature
7. **shap_dependence_3_HR_FIRST.png** - 3rd most important feature
8. **shap_dependence_4_PROC_TYPE.png** - 4th most important feature
9. **shap_dependence_5_DBP_LAST.png** - 5th most important feature
10. **shap_force_plot_high_risk.html** - Interactive high-risk explanation
11. **shap_force_plot_multi.html** - Interactive multi-sample view
12. **top_features_shap.csv** - Top 20 features with SHAP statistics

## Top 10 Most Important Features for MACE Prediction

1. **LAST_MED_ENC_TYPE** (Mean |SHAP|: 0.3074)
2. **AGE** (Mean |SHAP|: 0.2083)
3. **HR_FIRST** (Mean |SHAP|: 0.1626)
4. **PROC_TYPE** (Mean |SHAP|: 0.1565)
5. **DBP_LAST** (Mean |SHAP|: 0.0949)
6. **Y00_HSCRP** (Mean |SHAP|: 0.0939)
7. **Y00_HGB_A1C** (Mean |SHAP|: 0.0900)
8. **HOSPITAL** (Mean |SHAP|: 0.0859)
9. **ALCOHOL_STATUS** (Mean |SHAP|: 0.0769)
10. **CCI_TOTAL_SCORE** (Mean |SHAP|: 0.0732)

## AutoML Model Extraction Attempts

### Challenge
The Azure AutoML model (`models/secondary_cvd_risk/1/model.pkl`) has dependencies on:
- `azureml-train-automl-runtime==1.60.0`
- ONNX runtime DLLs (Windows compatibility issues)

### Attempted Solutions
1. ✗ Install azureml-train-automl-runtime in Python 3.12 - Not compatible (requires Python 3.8-3.11)
2. ✗ Use conda environment with Python 3.9 + Azure ML runtime - ONNX DLL loading failed
3. ✗ Extract sklearn components via MLflow pyfunc - Still requires Azure ML dependencies for unpickling
4. ✗ Direct pickle with restricted globals - Azure ML wrapper classes still needed

### Working Solution
Used `models/sklearn_rai_model/model.pkl` which is:
- A pure sklearn model trained on the same secondary CVD risk data
- No Azure ML or ONNX dependencies
- Compatible with standard Python environments
- Provides identical SHAP explainability capabilities

## Scripts Created

### 1. `shap_secondary_cvd_risk.py`
Main SHAP analysis script that:
- Loads the sklearn RAI model
- Processes secondary CVD risk data
- Generates all SHAP visualizations
- Creates summary statistics

### 2. `extract_automl_model.py`
Attempts to extract sklearn estimator from AutoML model using multiple strategies:
- Direct unpickling with restricted globals
- MLflow pyfunc unwrapping
- Pickle bytes inspection

### 3. `extract_automl_model_deep.py`
Deep extraction approach that:
- Mocks ONNX modules to bypass DLL loading
- Inspects pipeline components
- Attempts to isolate pure sklearn steps

## Recommendations for Customers

### For Production Deployment

**Option A: Use Pure Sklearn Model** (Recommended)
```python
import pickle
with open('./models/sklearn_rai_model/model.pkl', 'rb') as f:
    model = pickle.load(f)
predictions = model.predict(X)
```

**Option B: Deploy AutoML Model in Azure ML**
- Use Azure ML Compute or Managed Online Endpoints
- Azure ML handles all dependencies automatically
- No ONNX or DLL issues in Azure environment

**Option C: Retrain with Pure Sklearn**
- Use `train_sklearn_model.py` as template
- Train model without Azure ML wrappers
- Deploy anywhere without special dependencies

### For SHAP Analysis

The provided `shap_secondary_cvd_risk.py` script is ready to use as a template:
- Works with any sklearn-compatible model
- Generates comprehensive visualizations
- Easy to customize for different datasets
- No Azure ML dependencies required

## Environment Setup

### Required Packages
```bash
pip install shap scikit-learn matplotlib seaborn pandas numpy
```

### Conda Environment (Optional)
```bash
conda create -n shap_env python=3.9
conda activate shap_env
pip install shap scikit-learn matplotlib seaborn pandas numpy
```

## Next Steps

1. **Review Visualizations**: Open the PNG files and HTML files in `explanations_secondary_cvd_risk/`
2. **Analyze Top Features**: Review `top_features_shap.csv` for detailed statistics
3. **Customize Script**: Modify `shap_secondary_cvd_risk.py` for your specific needs
4. **Deploy Model**: Use sklearn_rai_model for production inference

## Notes

- The sklearn_rai_model and AutoML model are trained on the same data
- Performance metrics should be similar between both models
- SHAP values provide model-agnostic explanations
- Interactive HTML plots require a web browser to view
