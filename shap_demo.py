"""
Example: SHAP Explanations for Azure ML AutoML Models
This script demonstrates how to generate SHAP explanations for your trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Example with a simple dataset and model
def demo_shap_explanations():
    """Demonstrate SHAP explanations with a simple example"""
    
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_classification
    except ImportError as e:
        print(f"Required libraries not installed: {e}")
        print("Install with: pip install shap scikit-learn matplotlib seaborn")
        return
    
    print("=" * 60)
    print("SHAP Explanations Demo")
    print("=" * 60)
    
    # Create sample data (similar to healthcare data)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names similar to your healthcare dataset
    feature_names = [
        'age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'cholesterol', 'glucose', 'smoking_history', 'family_history',
        'exercise_frequency', 'medication_adherence'
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {feature_names}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Generate SHAP explanations
    print("\nGenerating SHAP explanations...")
    
    # Create output directory
    output_dir = Path("./explanations_demo")
    output_dir.mkdir(exist_ok=True)
    
    # For tree-based models, use TreeExplainer (much faster)
    # Use the new SHAP API that returns Explanation objects
    explainer = shap.TreeExplainer(model)
    
    # Get SHAP values using the new API (returns Explanation object)
    X_test_df = pd.DataFrame(X_test[:100], columns=feature_names)
    shap_values = explainer(X_test_df)
    
    # For binary classification with the new API, get values for positive class
    # shap_values.values has shape (n_samples, n_features, n_classes)
    if len(shap_values.values.shape) == 3:
        shap_values_pos = shap_values[:, :, 1]  # positive class
    else:
        shap_values_pos = shap_values
    
    # 1. Feature importance plot
    print("Creating feature importance plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_pos, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Summary plot (beeswarm)
    print("Creating summary plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values_pos, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall plot for first prediction
    print("Creating waterfall plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values_pos[0], show=False)
    plt.title('SHAP Waterfall Plot (First Sample)')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Dependence plot for most important feature
    # Get the most important feature index
    mean_abs_shap = np.abs(shap_values_pos.values).mean(0)
    most_important_idx = mean_abs_shap.argmax()
    most_important_feature = feature_names[most_important_idx]
    print(f"Creating dependence plot for: {most_important_feature}")
    
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(shap_values_pos[:, most_important_idx], show=False)
    plt.title(f'SHAP Dependence Plot: {most_important_feature}')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Force plot for individual prediction (save as HTML)
    print("Creating force plot...")
    base_value = shap_values_pos[0].base_values
    force_plot = shap.force_plot(
        base_value,
        shap_values_pos[0].values,
        X_test_df.iloc[0],
        feature_names=feature_names
    )
    shap.save_html(str(output_dir / 'shap_force_plot.html'), force_plot)
    
    print("\n" + "=" * 60)
    print("SHAP explanations generated successfully!")
    print(f"Results saved to: {output_dir}")
    print("Files created:")
    print("- shap_feature_importance.png: Overall feature importance")
    print("- shap_summary.png: Feature impact distribution")
    print("- shap_waterfall.png: Individual prediction breakdown")
    print("- shap_dependence.png: Feature interaction analysis")
    print("- shap_force_plot.html: Interactive individual prediction")
    print("=" * 60)


def example_usage_with_your_model():
    """Example code for using SHAP with your actual AutoML model"""
    
    example_code = '''
# Example: Using SHAP with your AutoML model from Azure ML

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

# 1. Connect to Azure ML and download your model
ml_client = MLClient(
    credential=AzureCliCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace"
)

# 2. Download the model (replace with your actual model name/version)
model_name = "secondary_cvd_risk_model"
model_version = "1"
model_path = ml_client.models.download(name=model_name, version=model_version, download_path="./models")

# 3. Load your preprocessed data
df = pd.read_csv("./data/preprocessed/your_preprocessed_data.csv")
feature_columns = [col for col in df.columns if col != 'mace']  # Replace 'mace' with your target column
X = df[feature_columns]
y = df['mace']  # Replace with your target column

# 4. Load the trained model (format depends on AutoML output)
import joblib
model = joblib.load("./models/model.pkl")  # Adjust path based on downloaded model

# 5. Create SHAP explainer
# For tree-based models (XGBoost, Random Forest, etc.)
explainer = shap.TreeExplainer(model)

# For other models, use general explainer (slower)
# explainer = shap.Explainer(model, X.sample(min(1000, len(X))))

# 6. Calculate SHAP values
sample_size = min(100, len(X))
X_sample = X.sample(sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)

# For binary classification, use positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Positive class

# 7. Create visualizations
# Feature importance
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.title('Feature Importance for CVD Risk Prediction')
plt.tight_layout()
plt.savefig('./explanations/cvd_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Summary plot
shap.summary_plot(shap_values, X_sample, show=False)
plt.title('SHAP Summary Plot for CVD Risk Prediction')
plt.tight_layout()
plt.savefig('./explanations/cvd_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Individual prediction explanation
patient_idx = 0  # Explain first patient
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[patient_idx], 
        base_values=explainer.expected_value,
        data=X_sample.iloc[patient_idx],
        feature_names=feature_columns
    ),
    show=False
)
plt.title(f'CVD Risk Explanation for Patient {patient_idx}')
plt.tight_layout()
plt.savefig('./explanations/cvd_patient_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

print("SHAP explanations generated for CVD risk model!")
'''
    
    print("=" * 60)
    print("Example Code for Your AutoML Model")
    print("=" * 60)
    print(example_code)
    print("=" * 60)


if __name__ == "__main__":
    print("Running SHAP demonstration...")
    demo_shap_explanations()
    
    print("\n\nFor your actual AutoML model, use this template:")
    example_usage_with_your_model()