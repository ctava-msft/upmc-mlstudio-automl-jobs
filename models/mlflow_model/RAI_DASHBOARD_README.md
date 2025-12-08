# Secondary CVD Risk Model - RAI Dashboard Configuration

This document describes the RAI (Responsible AI) dashboard compatibility configuration for the Secondary CVD Risk prediction model.

## Overview

The model is wrapped with `RAIModelWrapper`, a custom MLflow PythonModel that provides full compatibility with Azure ML Responsible AI dashboard components.

## Docker Images

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| `secondary-cvd-risk-model:latest` | `Dockerfile` | Basic inference |
| `secondary-cvd-risk-rai:latest` | `Dockerfile.rai` | RAI dashboard compatible |

### Build and Run

```bash
# Build RAI-compatible image
docker build -t secondary-cvd-risk-rai:latest -f Dockerfile.rai .

# Run container
docker run -d --rm --name secondary-cvd-rai -p 5001:5001 secondary-cvd-risk-rai:latest

# Test inference
python inference.py --output inference_results.txt
```

---

## RAI Dashboard Components Supported

| Component | Status | Description |
|-----------|--------|-------------|
| `rai_tabular_insight_constructor` | ✅ Supported | Initializes the RAI dashboard |
| `rai_tabular_erroranalysis` | ✅ Supported | Identifies error patterns and cohorts |
| `rai_tabular_explanation` | ✅ Supported | SHAP-based feature explanations |
| `rai_tabular_insight_gather` | ✅ Supported | Aggregates all insights for visualization |

---

## Model Metadata

### Task Configuration

| Property | Value |
|----------|-------|
| `model_type` | `classification` |
| `task_type` | `classification` |
| `task` | `binary_classification` |
| `target_column` | `CVD_RISK` |

### Class Information

| Property | Value |
|----------|-------|
| `classes` | `['No CVD Risk', 'CVD Risk']` |
| `classes_` (sklearn) | `np.array([0, 1])` |
| `n_classes` | `2` |

### Feature Information

| Property | Value |
|----------|-------|
| `n_features` | `110` |
| `feature_names_in_` | Array of 110 feature names |

### RAI Compatibility Flags

| Flag | Value |
|------|-------|
| `rai_compatible` | `True` |
| `supports_error_analysis` | `True` |
| `supports_explanations` | `True` |
| `supports_fairness_analysis` | `True` |
| `supports_model_overview` | `True` |

---

## RAIModelWrapper Class

### Required Attributes

```python
# Classification settings
model_type = 'classification'
task_type = 'classification'

# Class labels (integer for sklearn compatibility)
classes_ = np.array([0, 1])  # 0=No Risk, 1=Risk
class_names = ['No CVD Risk', 'CVD Risk']
n_classes_ = 2

# Feature information for explainability
feature_names_in_ = np.array([...])  # 110 feature names
n_features_in_ = 110

# Feature importances (if available from underlying model)
feature_importances_ = model.feature_importances_
```

### Required Methods

#### `predict(X)`
Returns integer class labels for RAI compatibility.

```python
def predict(self, ctx_or_data, model_input=None):
    """
    Returns: numpy array of integer class predictions (0=No Risk, 1=Risk)
    """
```

#### `predict_proba(X)`
Returns probability scores for each class - required for Error Analysis and Explanations.

```python
def predict_proba(self, data):
    """
    Returns: numpy array of shape (n_samples, 2)
        Column 0: P(No CVD Risk)
        Column 1: P(CVD Risk)
    """
```

#### `get_feature_names()`
Returns feature names for SHAP explanations.

```python
def get_feature_names(self):
    """Returns list of 110 feature names"""
```

#### `get_params(deep=True)`
sklearn compatibility method.

```python
def get_params(self, deep=True):
    """Returns model parameters dictionary"""
```

---

## Feature List (110 Features)

### Demographics (11 features)
- `AGE`, `SEX`, `RACE`, `RACE_LABEL`, `ETHNICITY`, `ETHNICITY_DETAILED`
- `ADMITDATE`, `PROC_DATE`, `DISCHARGEDATE`, `HOSPITAL`, `PROC_TYPE`

### Vitals (7 features)
- `BMI_IP`
- `SBP_FIRST`, `SBP_LAST`, `DBP_FIRST`, `DBP_LAST`
- `HR_FIRST`, `HR_LAST`

### Diagnoses (5 features)
- `CANCER_DX`, `ALZHEIMER_DX`, `NONSPECIFIC_MCI_DX`
- `VASCULAR_COGNITIVE_IMPAIRMENT_DX`, `NONSPECIFIC_COGNITIVE_DEFICIT_DX`

### Medical History (10 features)
- `CHF_HST`, `DIAB_HST`, `AFIB_HST`, `OBESE_HST`, `MORBIDOBESE_HST`
- `TIA_HST`, `CARDIOMYOPATHY_HST`
- `TOBACCO_STATUS`, `TOBACCO_STATUS_LABEL`
- `ALCOHOL_STATUS`, `ALCOHOL_STATUS_LABEL`
- `ILL_DRUG_STATUS`, `ILL_DRUG_STATUS_LABEL`

### Charlson Comorbidity Index (16 features)
- `CCI_CHF`, `CCI_PERIPHERAL_VASC`, `CCI_DEMENTIA`, `CCI_COPD`
- `CCI_RHEUMATIC_DISEASE`, `CCI_PEPTIC_ULCER`, `CCI_MILD_LIVER_DISEASE`
- `CCI_DM_NO_CC`, `CCI_DM_WITH_CC`, `CCI_HEMIPLEGIA`
- `CCI_RENAL_DISEASE`, `CCI_MALIG_NO_SKIN`, `CCI_SEVERE_LIVER_DISEASE`
- `CCI_METASTATIC_TUMOR`, `CCI_AIDS_HIV`, `CCI_TOTAL_SCORE`

### Elixhauser Comorbidity (32 features)
- Cardiac: `ELIX_CARDIAC_ARRTHYTHMIAS`, `ELIX_CONGESTIVE_HEART_FAILURE`, `ELIX_VALVULAR_DISEASE`
- Vascular: `ELIX_PULM_CIRC_DISORDERS`, `ELIX_PERIPH_VASC_DISEASE`, `ELIX_HYPERTENSION`
- Neurological: `ELIX_PARALYSIS`, `ELIX_NEURO_DISORDERS`
- Metabolic: `ELIX_DIABETES_WO_CC`, `ELIX_DIABETES_W_CC`, `ELIX_HYPOTHYROIDISM`, `ELIX_OBESITY`
- Other: `ELIX_COPD`, `ELIX_RENAL_FAILURE`, `ELIX_LIVER_DISEASE`, etc.
- Scores: `ELIX_AHRQ_SCORE`, `ELIX_VAN_WALRAVEN_SCORE`

### Current Medications (19 features)
- `MED_CURRENT_ASA`, `MED_CURRENT_STATIN`, `MED_CURRENT_LOW_STATIN`
- `MED_CURRENT_MODERATE_STATIN`, `MED_CURRENT_HIGH_STATIN`
- `MED_CURRENT_BB`, `MED_CURRENT_AB`, `MED_CURRENT_CCB`, `MED_CURRENT_ARB`
- `MED_CURRENT_ZETIA`, `MED_CURRENT_PCSK9`, `MED_CURRENT_WARFARIN`
- `MED_CURRENT_DOAC`, `MED_CURRENT_COLCHICINE`, `MED_CURRENT_ARNI`
- `MED_CURRENT_HYDRALAZINE`, `MED_CURRENT_MRA`, `MED_CURRENT_SPIRONOLACTONE`
- `MED_CURRENT_MEMORY_AGENT`

### Lab Values (6 features)
- `Y00_HGB_A1C`, `Y00_TRIGLYCERIDE`, `Y00_HDL`
- `Y00_LDL`, `Y00_CHOLESTEROL`, `Y00_HSCRP`

### Encounter (1 feature)
- `LAST_MED_ENC_TYPE`

---

## Azure ML Registration

To register this model for use with RAI dashboard in Azure ML:

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Connect to workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# Register as MLFLOW_MODEL for RAI dashboard
model = Model(
    path="models/secondary_cvd_risk/1",
    type=AssetTypes.MLFLOW_MODEL,
    name="secondary-cvd-risk-rai",
    description="Secondary CVD Risk model with RAI dashboard support",
)

registered_model = ml_client.models.create_or_update(model)
print(f"Model registered: {registered_model.name}:{registered_model.version}")
```

---

## RAI Dashboard Pipeline

Example pipeline to create RAI dashboard:

```python
from azure.ai.ml import Input, dsl
from azure.ai.ml.constants import AssetTypes

# Get RAI components from registry
ml_client_registry = MLClient(
    credential=credential,
    subscription_id=ml_client.subscription_id,
    resource_group_name=ml_client.resource_group_name,
    registry_name="azureml",
)

rai_constructor = ml_client_registry.components.get(
    name="rai_tabular_insight_constructor", label="latest"
)
version = rai_constructor.version

rai_erroranalysis = ml_client_registry.components.get(
    name="rai_tabular_erroranalysis", version=version
)
rai_explanation = ml_client_registry.components.get(
    name="rai_tabular_explanation", version=version
)
rai_gather = ml_client_registry.components.get(
    name="rai_tabular_insight_gather", version=version
)

@dsl.pipeline(compute="aml-cluster", description="RAI insights on CVD risk data")
def rai_cvd_pipeline(target_column_name, train_data, test_data):
    # Initialize RAI dashboard
    create_rai = rai_constructor(
        title="RAI Dashboard - Secondary CVD Risk",
        task_type="classification",
        model_info=expected_model_id,
        model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
        train_dataset=train_data,
        test_dataset=test_data,
        target_column_name=target_column_name,
    )
    
    # Add error analysis
    error_analysis = rai_erroranalysis(
        rai_insights_dashboard=create_rai.outputs.rai_insights_dashboard,
    )
    
    # Add explanations
    explanations = rai_explanation(
        rai_insights_dashboard=create_rai.outputs.rai_insights_dashboard,
        comment="SHAP feature explanations",
    )
    
    # Gather all insights
    gather = rai_gather(
        constructor=create_rai.outputs.rai_insights_dashboard,
        insight_3=error_analysis.outputs.error_analysis,
        insight_4=explanations.outputs.explanation,
    )
    
    return {"dashboard": gather.outputs.dashboard}
```

---

## Inference Results

The model returns integer class labels:

| Prediction | Meaning |
|------------|---------|
| `0` | No CVD Risk |
| `1` | CVD Risk |

### Example Output

```
Low Risk Patient Prediction:  0
High Risk Patient Prediction: 1
✅ Model correctly identified higher risk in the high-risk patient
```

---

## References

- [Azure ML Responsible AI Dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard)
- [RAI Dashboard Tutorial](https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/10/Create%20Responsible%20AI%20dashboard.ipynb)
- [MLflow Model Logging](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-mlflow-models)
