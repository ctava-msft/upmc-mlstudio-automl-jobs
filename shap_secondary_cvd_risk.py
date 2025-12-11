"""
SHAP Explanations for Secondary CVD Risk Model
This script generates SHAP explanations for the sklearn model trained on secondary CVD risk data.
Uses MLflow to load the AutoML model (handles AzureML dependencies automatically).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def run_shap_analysis():
    """Run SHAP analysis on the secondary CVD risk sklearn model"""
    
    try:
        import shap
        from sklearn.model_selection import train_test_split
        import mlflow
    except ImportError as e:
        print(f"Required libraries not installed: {e}")
        print("Install with: pip install shap scikit-learn matplotlib seaborn mlflow")
        return
    
    print("=" * 70)
    print("SHAP Explanations for Secondary CVD Risk Model (AutoML)")
    print("=" * 70)
    
    # Paths - Using the AutoML-generated model directory (for MLflow loading)
    model_dir = Path("./models/secondary_cvd_risk/1")
    # Use the original data source
    data_path = Path("./data/secondary_cvd_risk_min/secondary-cvd-risk.csv")
    output_dir = Path("./explanations_secondary_cvd_risk")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Data path: {data_path}")
    
    # Load the trained AutoML model using MLflow
    print("\nLoading AutoML-trained sklearn model via MLflow...")
    try:
        # MLflow can load the model using the sklearn flavor, which avoids the azureml.training dependency
        model = mlflow.sklearn.load_model(str(model_dir))
    except Exception as e:
        print(f"MLflow sklearn loading failed: {e}")
        print("Trying pyfunc loader...")
        try:
            model_pyfunc = mlflow.pyfunc.load_model(str(model_dir))
            # Get the underlying sklearn model
            model = model_pyfunc._model_impl.sklearn_model
        except Exception as e2:
            print(f"PyFunc loading also failed: {e2}")
            print("\nNote: The AutoML model has dependencies on azureml-train-automl-runtime.")
            print("You may need to install: pip install azureml-train-automl-runtime==1.60.0")
            return
    print(f"Model type: {type(model).__name__}")
    
    # Load the data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Target column
    target_column = 'MACE'
    
    # Columns to drop (non-feature columns)
    columns_to_drop = [
        target_column,
        'RACE_LABEL', 'ETHNICITY_DETAILED', 'ADMITDATE', 'PROC_DATE', 
        'DISCHARGEDATE', 'TOBACCO_STATUS_LABEL', 'ALCOHOL_STATUS_LABEL',
        'ILL_DRUG_STATUS_LABEL', 'LAST_MED_ENC_TYPE'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # Prepare features - drop non-feature columns
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]
    
    # Handle categorical columns - encode them for the model
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {categorical_cols}")
    
    # Encode categorical columns using label encoding
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('UNKNOWN').astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Fill numeric NaN values with 0
    X = X.fillna(0)
    
    # Get feature names from the processed data
    feature_names = list(X.columns)
    print(f"Number of features: {len(feature_names)}")
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Sample data for SHAP (use a subset for efficiency)
    sample_size = min(500, len(X))
    print(f"\nSampling {sample_size} instances for SHAP analysis...")
    
    # Stratified sampling to maintain class balance
    np.random.seed(42)
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]
    
    n_class_1 = min(len(idx_class_1), sample_size // 4)  # ~25% positive class
    n_class_0 = min(len(idx_class_0), sample_size - n_class_1)
    
    sample_idx = np.concatenate([
        np.random.choice(idx_class_0, n_class_0, replace=False),
        np.random.choice(idx_class_1, n_class_1, replace=False)
    ])
    np.random.shuffle(sample_idx)
    
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    y_sample = y.iloc[sample_idx].reset_index(drop=True)
    
    print(f"Sample size: {len(X_sample)}")
    print(f"Sample class distribution: {y_sample.value_counts().to_dict()}")
    
    # Generate SHAP explanations
    print("\n" + "=" * 70)
    print("Generating SHAP Explanations...")
    print("=" * 70)
    
    # Create explainer - use TreeExplainer for tree-based models
    print("\nCreating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values using the new API
    print("Calculating SHAP values (this may take a few minutes)...")
    shap_values = explainer(X_sample)
    
    # For binary classification, get positive class SHAP values
    if len(shap_values.values.shape) == 3:
        shap_values_pos = shap_values[:, :, 1]
        print("Using positive class (MACE=1) SHAP values")
    else:
        shap_values_pos = shap_values
    
    # =========================================================================
    # 1. Feature Importance Plot (Bar)
    # =========================================================================
    print("\n1. Creating feature importance plot...")
    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_values_pos, max_display=30, show=False)
    plt.title('SHAP Feature Importance for Secondary CVD Risk (MACE) Prediction\n(Top 30 Features)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir / 'shap_feature_importance.png'}")
    
    # =========================================================================
    # 2. Summary Plot (Beeswarm)
    # =========================================================================
    print("2. Creating summary beeswarm plot...")
    plt.figure(figsize=(12, 12))
    shap.plots.beeswarm(shap_values_pos, max_display=30, show=False)
    plt.title('SHAP Summary Plot: Feature Impact on MACE Prediction\n(Top 30 Features)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir / 'shap_summary_beeswarm.png'}")
    
    # =========================================================================
    # 3. Waterfall Plot for High-Risk Patient
    # =========================================================================
    print("3. Creating waterfall plot for high-risk patient...")
    # Find a patient with MACE=1 (positive outcome)
    positive_idx = np.where(y_sample == 1)[0]
    if len(positive_idx) > 0:
        patient_idx = positive_idx[0]
        patient_label = "High-Risk (MACE=1)"
    else:
        patient_idx = 0
        patient_label = "Sample Patient"
    
    plt.figure(figsize=(12, 10))
    shap.plots.waterfall(shap_values_pos[patient_idx], max_display=20, show=False)
    plt.title(f'SHAP Waterfall Plot: {patient_label}\nIndividual Prediction Breakdown', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_waterfall_high_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir / 'shap_waterfall_high_risk.png'}")
    
    # =========================================================================
    # 4. Waterfall Plot for Low-Risk Patient
    # =========================================================================
    print("4. Creating waterfall plot for low-risk patient...")
    negative_idx = np.where(y_sample == 0)[0]
    if len(negative_idx) > 0:
        patient_idx_low = negative_idx[0]
        patient_label_low = "Low-Risk (MACE=0)"
    else:
        patient_idx_low = 1
        patient_label_low = "Sample Patient"
    
    plt.figure(figsize=(12, 10))
    shap.plots.waterfall(shap_values_pos[patient_idx_low], max_display=20, show=False)
    plt.title(f'SHAP Waterfall Plot: {patient_label_low}\nIndividual Prediction Breakdown', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_waterfall_low_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir / 'shap_waterfall_low_risk.png'}")
    
    # =========================================================================
    # 5. Dependence Plots for Top Features
    # =========================================================================
    print("5. Creating dependence plots for top 5 features...")
    mean_abs_shap = np.abs(shap_values_pos.values).mean(0)
    top_feature_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    
    for i, feat_idx in enumerate(top_feature_indices):
        feat_name = feature_names[feat_idx]
        plt.figure(figsize=(10, 6))
        shap.plots.scatter(shap_values_pos[:, feat_idx], show=False)
        plt.title(f'SHAP Dependence Plot: {feat_name}\n(Feature #{i+1} by Importance)', 
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        safe_name = feat_name.replace('/', '_').replace(' ', '_')
        plt.savefig(output_dir / f'shap_dependence_{i+1}_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: shap_dependence_{i+1}_{safe_name}.png")
    
    # =========================================================================
    # 6. Force Plot (Interactive HTML)
    # =========================================================================
    print("6. Creating interactive force plot...")
    base_value = shap_values_pos[patient_idx].base_values
    force_plot = shap.force_plot(
        base_value,
        shap_values_pos[patient_idx].values,
        X_sample.iloc[patient_idx],
        feature_names=feature_names
    )
    shap.save_html(str(output_dir / 'shap_force_plot_high_risk.html'), force_plot)
    print(f"   Saved: {output_dir / 'shap_force_plot_high_risk.html'}")
    
    # Force plot for multiple samples
    print("   Creating multi-sample force plot...")
    # Use the same base value we already extracted
    force_plot_multi = shap.force_plot(
        base_value,
        shap_values_pos[:50].values,
        X_sample.iloc[:50],
        feature_names=feature_names
    )
    shap.save_html(str(output_dir / 'shap_force_plot_multi.html'), force_plot_multi)
    print(f"   Saved: {output_dir / 'shap_force_plot_multi.html'}")
    
    # =========================================================================
    # 7. Create Summary Statistics Report
    # =========================================================================
    print("7. Creating summary statistics report...")
    
    # Get top 20 features by importance
    top_20_idx = np.argsort(mean_abs_shap)[-20:][::-1]
    top_features_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_20_idx],
        'Mean |SHAP|': mean_abs_shap[top_20_idx],
        'Mean SHAP': shap_values_pos.values[:, top_20_idx].mean(0),
        'Std SHAP': shap_values_pos.values[:, top_20_idx].std(0)
    })
    top_features_df.to_csv(output_dir / 'top_features_shap.csv', index=False)
    print(f"   Saved: {output_dir / 'top_features_shap.csv'}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SHAP Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles created:")
    print("  1. shap_feature_importance.png     - Overall feature importance (bar chart)")
    print("  2. shap_summary_beeswarm.png       - Feature impact distribution")
    print("  3. shap_waterfall_high_risk.png    - High-risk patient breakdown")
    print("  4. shap_waterfall_low_risk.png     - Low-risk patient breakdown")
    print("  5. shap_dependence_*.png           - Top 5 feature dependence plots")
    print("  6. shap_force_plot_high_risk.html  - Interactive high-risk explanation")
    print("  7. shap_force_plot_multi.html      - Interactive multi-sample view")
    print("  8. top_features_shap.csv           - Top 20 features with SHAP stats")
    
    print("\n" + "=" * 70)
    print("Top 10 Most Important Features for MACE Prediction:")
    print("=" * 70)
    for i, row in top_features_df.head(10).iterrows():
        direction = "↑" if row['Mean SHAP'] > 0 else "↓"
        print(f"  {i+1:2}. {row['Feature']:40} | Mean |SHAP|: {row['Mean |SHAP|']:.4f} {direction}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_shap_analysis()
