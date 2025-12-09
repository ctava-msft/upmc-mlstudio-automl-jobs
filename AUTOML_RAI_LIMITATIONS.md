# AutoML and Responsible AI (RAI) Dashboard: Findings and Limitations

## Executive Summary

**AutoML models are NOT compatible with the standalone RAI Dashboard pipeline components.** This is a documented limitation in Azure Machine Learning, not a bug.

**CONFIRMED: December 8, 2025** - Multiple wrapper approaches were attempted, but all fail because the internal AutoML model structure contains Azure ML SDK internal classes that cannot be deserialized in the RAI environment.

---

## The Problem

When attempting to run RAI dashboard pipeline jobs on AutoML-trained models, the following error occurs:

```
ModuleNotFoundError: No module named 'azureml._base_sdk_common._docstring_wrapper'
```

Or related variants like:
```
No module named 'azureml._base_sdk_common._docstring_wrapper'; 'azureml._base_sdk_common' is not a package
No module named 'wrapper'
```

### Root Cause

AutoML models are serialized (pickled) with internal Azure ML SDK classes that:
1. Are part of the private `azureml._base_sdk_common` package
2. Reference internal decorators like `_docstring_wrapper`
3. May change between SDK versions without notice
4. Get embedded into the pickled model during training

When RAI components try to load the model in their environment (which has different/newer Azure ML packages), Python's pickle module fails because the exact class structure isn't available.

### Why Wrappers Don't Work

Multiple wrapper approaches were attempted:
1. **Sklearn-compatible wrapper** - Still needs to `pickle.load(model.pkl)` which triggers the internal class issue
2. **MLflow pyfunc wrapper** - Same problem - the underlying model.pkl contains incompatible classes
3. **Custom environment** - RAI components use their own fixed environment, not custom ones

The fundamental issue is that `model.pkl` from AutoML contains internal Azure ML classes that don't exist in the RAI runtime.

---

## Official Microsoft Documentation

### From [Responsible AI Dashboard Limitations](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?view=azureml-api-2#supported-scenarios-and-limitations):

> **"The Responsible AI dashboard currently doesn't support the AutoML MLFlow model."**
>
> **"The Responsible AI dashboard currently doesn't support registered AutoML models from the UI."**

### From [Generate RAI Insights with SDK/CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-insights-sdk-cli?view=azureml-api-2):

> **"All models must be registered in Azure Machine Learning in MLflow format with a sklearn (scikit-learn) flavor."**
>
> **"The models must be loadable in the component environment."**
>
> **"The models must be pickleable."**

---

## GitHub Issues

**No specific GitHub issue exists** for the `azureml._base_sdk_common._docstring_wrapper` error. Searches across:
- `Azure/azure-sdk-for-python`
- `Azure/MachineLearningNotebooks`
- `Azure/azureml-examples`
- `microsoft/responsible-ai-toolbox`

...all returned **zero results**. This is because it's a documented limitation, not a bug to be fixed.

---

## Version Information

### Model Training Environment
The model was trained with:
```yaml
# From conda.yaml
azureml-train-automl-runtime==1.60.0
azureml-interpret==1.60.0
azureml-defaults==1.60.0
scikit-learn==1.5.1
```

### RAI Component Environment
RAI components run in a different environment with potentially different versions of:
- `azureml-core`
- `azureml-mlflow`
- Internal Azure ML packages

### The Mismatch
| Aspect | AutoML SDK (1.60.0) | RAI Component Environment |
|--------|---------------------|--------------------------|
| Internal module | `azureml._base_sdk_common._docstring_wrapper` exists | Module doesn't exist or has different structure |
| Model type | AutoML pipeline with preprocessing | Expects pure sklearn model |
| Pickle references | Contains internal AutoML decorators/wrappers | Can't unpickle these references |

---

## Supported Scenarios

### What RAI Dashboard DOES Support
| Feature | Supported |
|---------|-----------|
| Regression models | ✅ Yes |
| Classification (binary) | ✅ Yes |
| Classification (multi-class) | ✅ Yes |
| Tabular data | ✅ Yes |
| MLflow models with sklearn flavor | ✅ Yes |
| Pure scikit-learn models | ✅ Yes |
| Models implementing `predict()`/`predict_proba()` | ✅ Yes |

### What RAI Dashboard Does NOT Support
| Feature | Supported |
|---------|-----------|
| **AutoML MLflow models** | ❌ No |
| **Registered AutoML models from UI** | ❌ No |
| Forecasting models (certain algorithms) | ❌ No |
| Image/Vision models (in Azure ML) | ❌ No |
| Text/NLP models (in Azure ML) | ❌ No |
| Datasets > 10,000 columns | ❌ No |
| NumPy/SciPy sparse data | ❌ No |

---

## Solutions and Workarounds

### ✅ Option 1: Use AutoML's Built-in RAI (Recommended)

Generate the RAI dashboard **during** AutoML training, not after.

**Requirements:**
1. `enable_model_explainability=True` in job configuration
2. **Serverless compute** (not a named compute cluster)

**Code Example:**
```python
from azure.ai.ml import automl
from azure.ai.ml.entities import ResourceConfiguration

classification_job = automl.classification(
    # Do NOT specify compute= parameter
    experiment_name="my-experiment",
    training_data=training_data,
    target_column_name="target",
    enable_model_explainability=True,  # Required for RAI
)

# Use serverless compute - REQUIRED for RAI dashboard generation
classification_job.resources = ResourceConfiguration(
    instance_type="Standard_E4s_v3",
    instance_count=1
)
```

**Limitation:** 
> "Responsible AI dashboards can't be generated for an existing Automated ML model. The dashboard is created only for the best recommended model when you create a **new** Automated ML job."

### ⚠️ Option 2: Train a Pure Sklearn Model

If you need the full RAI pipeline flexibility, train a model using pure scikit-learn:

```python
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn

# Train a pure sklearn model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log with MLflow
mlflow.sklearn.log_model(model, "model")
```

This model will be fully compatible with RAI dashboard components.

### ⚠️ Option 3: Create a Sklearn Wrapper

Wrap your AutoML model's prediction logic in a pure sklearn-compatible class:

```python
class SklearnCompatibleWrapper:
    def __init__(self, predict_fn, predict_proba_fn=None):
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
    
    def predict(self, X):
        return self.predict_fn(X)
    
    def predict_proba(self, X):
        if self.predict_proba_fn:
            return self.predict_proba_fn(X)
        raise NotImplementedError("predict_proba not available")
```

**Note:** This approach requires that you can call the AutoML model without unpickling its internal structure.

---

## RAI Dashboard Components

When using the built-in AutoML RAI, you get these components automatically:

| Component | Description |
|-----------|-------------|
| **Error Analysis** | Understand how model failures are distributed |
| **Model Overview & Fairness** | Performance metrics across cohorts |
| **Model Explanations** | Feature importance (global and local) |
| **Data Analysis** | Dataset exploration and statistics |

To get additional components (Causal Analysis, Counterfactuals), you would need a pure sklearn model.

---

## Accessing the AutoML RAI Dashboard

After running an AutoML job with serverless compute and `enable_model_explainability=True`:

1. Go to **Azure ML Studio**
2. Navigate to **Jobs** → Find your AutoML job
3. Click on the **Models** tab
4. Select the **best model**
5. View the **Responsible AI dashboard** tab

---

## References

1. [Assess AI systems using the Responsible AI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?view=azureml-api-2)
2. [Generate Responsible AI insights with SDK/CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-insights-sdk-cli?view=azureml-api-2)
3. [Set up AutoML training in studio UI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-automated-ml-for-ml-models?view=azureml-api-2)
4. [Evaluate AutoML experiment results](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml?view=azureml-api-2)
5. [Azure ML RAI Examples on GitHub](https://github.com/Azure/azureml-examples/tree/main/sdk/python/responsible-ai)

---

## Summary

| Approach | Works with AutoML? | Full RAI Features? | Effort |
|----------|-------------------|-------------------|--------|
| AutoML + Serverless + Built-in RAI | ✅ Yes | Partial (4 components) | Low |
| Standalone RAI Pipeline on AutoML model | ❌ No | N/A | N/A |
| Pure sklearn model + RAI Pipeline | ✅ Yes | Full (6 components) | High (retrain) |
| Sklearn wrapper | ⚠️ Maybe | Full | Medium |

**Recommendation:** Use AutoML's built-in RAI dashboard generation with serverless compute for the fastest path to responsible AI insights on AutoML models.
