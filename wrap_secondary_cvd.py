"""
Wrap the secondary_cvd_risk model for Azure ML Responsible AI Dashboard
and save outputs to a text file.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import rai_model_wrapper
sys.path.insert(0, str(Path(__file__).parent))

from rai_model_wrapper import (
    ModelWrapper,
    save_model_with_wrapper,
    get_conda_env_dict,
    save_conda_env
)


def load_secondary_cvd_model():
    """Load the secondary CVD risk model from the models directory."""
    model_path = Path(__file__).parent / "models" / "secondary_cvd_risk" / "1" / "model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def get_secondary_cvd_feature_names():
    """Get the expected feature names for the secondary CVD risk model."""
    return [
        'AGE', 'SEX', 'RACE', 'RACE_LABEL', 'ETHNICITY', 'ETHNICITY_DETAILED',
        'ADMITDATE', 'PROC_DATE', 'DISCHARGEDATE', 'HOSPITAL', 'PROC_TYPE',
        'BMI_IP', 'SBP_FIRST', 'SBP_LAST', 'DBP_FIRST', 'DBP_LAST', 'HR_FIRST', 'HR_LAST',
        'CANCER_DX', 'ALZHEIMER_DX', 'NONSPECIFIC_MCI_DX', 'VASCULAR_COGNITIVE_IMPAIRMENT_DX',
        'NONSPECIFIC_COGNITIVE_DEFICIT_DX', 'CHF_HST', 'DIAB_HST', 'AFIB_HST', 'OBESE_HST',
        'MORBIDOBESE_HST', 'TIA_HST', 'CARDIOMYOPATHY_HST', 'TOBACCO_STATUS', 'TOBACCO_STATUS_LABEL',
        'ALCOHOL_STATUS', 'ALCOHOL_STATUS_LABEL', 'ILL_DRUG_STATUS', 'ILL_DRUG_STATUS_LABEL',
        'CCI_CHF', 'CCI_PERIPHERAL_VASC', 'CCI_DEMENTIA', 'CCI_COPD', 'CCI_RHEUMATIC_DISEASE',
        'CCI_PEPTIC_ULCER', 'CCI_MILD_LIVER_DISEASE', 'CCI_DM_NO_CC', 'CCI_DM_WITH_CC',
        'CCI_HEMIPLEGIA', 'CCI_RENAL_DISEASE', 'CCI_MALIG_NO_SKIN', 'CCI_SEVERE_LIVER_DISEASE',
        'CCI_METASTATIC_TUMOR', 'CCI_AIDS_HIV', 'CCI_TOTAL_SCORE',
        'ELIX_CARDIAC_ARRTHYTHMIAS', 'ELIX_CONGESTIVE_HEART_FAILURE', 'ELIX_VALVULAR_DISEASE',
        'ELIX_PULM_CIRC_DISORDERS', 'ELIX_PERIPH_VASC_DISEASE', 'ELIX_HYPERTENSION',
        'ELIX_PARALYSIS', 'ELIX_NEURO_DISORDERS', 'ELIX_COPD', 'ELIX_DIABETES_WO_CC',
        'ELIX_DIABETES_W_CC', 'ELIX_HYPOTHYROIDISM', 'ELIX_RENAL_FAILURE', 'ELIX_LIVER_DISEASE',
        'ELIX_CHRONIC_PEPTIC_ULCER_DISEASE', 'ELIX_HIV_AIDS', 'ELIX_LYMPHOMA',
        'ELIX_METASTATIC_CANCER', 'ELIX_TUMOR_WO_METASTATIC_CANCER', 'ELIX_RHEUMATOID_ARTHRITIS',
        'ELIX_COAGULATION_DEFICIENCY', 'ELIX_OBESITY', 'ELIX_WEIGHT_LOSS',
        'ELIX_FLUID_ELECTROLYTE_DISORDERS', 'ELIX_ANEMIA_BLOOD_LOSS', 'ELIX_DEFICIENCY_ANEMIAS',
        'ELIX_ALCOHOL_ABUSE', 'ELIX_DRUG_ABUSE', 'ELIX_PSYCHOSES', 'ELIX_DEPRESSION',
        'ELIX_AHRQ_SCORE', 'ELIX_VAN_WALRAVEN_SCORE',
        'MED_CURRENT_ASA', 'MED_CURRENT_STATIN', 'MED_CURRENT_LOW_STATIN', 'MED_CURRENT_MODERATE_STATIN',
        'MED_CURRENT_HIGH_STATIN', 'MED_CURRENT_BB', 'MED_CURRENT_AB', 'MED_CURRENT_CCB',
        'MED_CURRENT_ARB', 'MED_CURRENT_ZETIA', 'MED_CURRENT_PCSK9', 'MED_CURRENT_WARFARIN',
        'MED_CURRENT_DOAC', 'MED_CURRENT_COLCHICINE', 'MED_CURRENT_ARNI', 'MED_CURRENT_HYDRALAZINE',
        'MED_CURRENT_MRA', 'MED_CURRENT_SPIRONOLACTONE', 'MED_CURRENT_MEMORY_AGENT',
        'Y00_HGB_A1C', 'Y00_TRIGLYCERIDE', 'Y00_HDL', 'Y00_LDL', 'Y00_CHOLESTEROL', 'Y00_HSCRP',
        'LAST_MED_ENC_TYPE'
    ]


def create_sample_patient_data():
    """Create sample patient data for testing."""
    # Low risk patient
    low_risk = {
        "AGE": 45, "SEX": 0, "RACE": 2, "RACE_LABEL": "White",
        "ETHNICITY": 1, "ETHNICITY_DETAILED": 1,
        "ADMITDATE": "2023-01-15", "PROC_DATE": "2023-01-16", "DISCHARGEDATE": "2023-01-20",
        "HOSPITAL": "Main", "PROC_TYPE": 1,
        "BMI_IP": 22.0, "SBP_FIRST": 120, "SBP_LAST": 118,
        "DBP_FIRST": 78, "DBP_LAST": 76, "HR_FIRST": 70, "HR_LAST": 68,
        "CANCER_DX": 0, "ALZHEIMER_DX": 0, "NONSPECIFIC_MCI_DX": 0,
        "VASCULAR_COGNITIVE_IMPAIRMENT_DX": 0, "NONSPECIFIC_COGNITIVE_DEFICIT_DX": 0,
        "CHF_HST": 0, "DIAB_HST": 0, "AFIB_HST": 0, "OBESE_HST": 0,
        "MORBIDOBESE_HST": 0, "TIA_HST": 0, "CARDIOMYOPATHY_HST": 0,
        "TOBACCO_STATUS": 0, "TOBACCO_STATUS_LABEL": "Non-smoker",
        "ALCOHOL_STATUS": 0, "ALCOHOL_STATUS_LABEL": "Non-drinker",
        "ILL_DRUG_STATUS": 0, "ILL_DRUG_STATUS_LABEL": "None",
        "CCI_CHF": 0, "CCI_PERIPHERAL_VASC": 0, "CCI_DEMENTIA": 0, "CCI_COPD": 0,
        "CCI_RHEUMATIC_DISEASE": 0, "CCI_PEPTIC_ULCER": 0, "CCI_MILD_LIVER_DISEASE": 0,
        "CCI_DM_NO_CC": 0, "CCI_DM_WITH_CC": 0, "CCI_HEMIPLEGIA": 0,
        "CCI_RENAL_DISEASE": 0, "CCI_MALIG_NO_SKIN": 0, "CCI_SEVERE_LIVER_DISEASE": 0,
        "CCI_METASTATIC_TUMOR": 0, "CCI_AIDS_HIV": 0, "CCI_TOTAL_SCORE": 0,
        "ELIX_CARDIAC_ARRTHYTHMIAS": 0, "ELIX_CONGESTIVE_HEART_FAILURE": 0,
        "ELIX_VALVULAR_DISEASE": 0, "ELIX_PULM_CIRC_DISORDERS": 0,
        "ELIX_PERIPH_VASC_DISEASE": 0, "ELIX_HYPERTENSION": 0, "ELIX_PARALYSIS": 0,
        "ELIX_NEURO_DISORDERS": 0, "ELIX_COPD": 0, "ELIX_DIABETES_WO_CC": 0,
        "ELIX_DIABETES_W_CC": 0, "ELIX_HYPOTHYROIDISM": 0, "ELIX_RENAL_FAILURE": 0,
        "ELIX_LIVER_DISEASE": 0, "ELIX_CHRONIC_PEPTIC_ULCER_DISEASE": 0, "ELIX_HIV_AIDS": 0,
        "ELIX_LYMPHOMA": 0, "ELIX_METASTATIC_CANCER": 0, "ELIX_TUMOR_WO_METASTATIC_CANCER": 0,
        "ELIX_RHEUMATOID_ARTHRITIS": 0, "ELIX_COAGULATION_DEFICIENCY": 0, "ELIX_OBESITY": 0,
        "ELIX_WEIGHT_LOSS": 0, "ELIX_FLUID_ELECTROLYTE_DISORDERS": 0,
        "ELIX_ANEMIA_BLOOD_LOSS": 0, "ELIX_DEFICIENCY_ANEMIAS": 0,
        "ELIX_ALCOHOL_ABUSE": 0, "ELIX_DRUG_ABUSE": 0, "ELIX_PSYCHOSES": 0, "ELIX_DEPRESSION": 0,
        "ELIX_AHRQ_SCORE": 0, "ELIX_VAN_WALRAVEN_SCORE": 0,
        "MED_CURRENT_ASA": 0, "MED_CURRENT_STATIN": 0, "MED_CURRENT_LOW_STATIN": 0,
        "MED_CURRENT_MODERATE_STATIN": 0, "MED_CURRENT_HIGH_STATIN": 0,
        "MED_CURRENT_BB": 0, "MED_CURRENT_AB": 0, "MED_CURRENT_CCB": 0,
        "MED_CURRENT_ARB": 0, "MED_CURRENT_ZETIA": 0, "MED_CURRENT_PCSK9": 0,
        "MED_CURRENT_WARFARIN": 0, "MED_CURRENT_DOAC": 0, "MED_CURRENT_COLCHICINE": 0,
        "MED_CURRENT_ARNI": 0, "MED_CURRENT_HYDRALAZINE": 0, "MED_CURRENT_MRA": 0,
        "MED_CURRENT_SPIRONOLACTONE": 0, "MED_CURRENT_MEMORY_AGENT": 0,
        "Y00_HGB_A1C": 5.2, "Y00_TRIGLYCERIDE": 90, "Y00_HDL": 60,
        "Y00_LDL": 95, "Y00_CHOLESTEROL": 180, "Y00_HSCRP": 0.8,
        "LAST_MED_ENC_TYPE": "Outpatient"
    }
    
    # High risk patient
    high_risk = {
        "AGE": 72, "SEX": 1, "RACE": 2, "RACE_LABEL": "White",
        "ETHNICITY": 1, "ETHNICITY_DETAILED": 1,
        "ADMITDATE": "2023-01-15", "PROC_DATE": "2023-01-16", "DISCHARGEDATE": "2023-01-20",
        "HOSPITAL": "Main", "PROC_TYPE": 1,
        "BMI_IP": 32.5, "SBP_FIRST": 165, "SBP_LAST": 160,
        "DBP_FIRST": 95, "DBP_LAST": 90, "HR_FIRST": 85, "HR_LAST": 82,
        "CANCER_DX": 0, "ALZHEIMER_DX": 0, "NONSPECIFIC_MCI_DX": 0,
        "VASCULAR_COGNITIVE_IMPAIRMENT_DX": 0, "NONSPECIFIC_COGNITIVE_DEFICIT_DX": 0,
        "CHF_HST": 1, "DIAB_HST": 1, "AFIB_HST": 1, "OBESE_HST": 1,
        "MORBIDOBESE_HST": 1, "TIA_HST": 1, "CARDIOMYOPATHY_HST": 1,
        "TOBACCO_STATUS": 2, "TOBACCO_STATUS_LABEL": "Current smoker",
        "ALCOHOL_STATUS": 1, "ALCOHOL_STATUS_LABEL": "Social drinker",
        "ILL_DRUG_STATUS": 0, "ILL_DRUG_STATUS_LABEL": "None",
        "CCI_CHF": 1, "CCI_PERIPHERAL_VASC": 1, "CCI_DEMENTIA": 0, "CCI_COPD": 1,
        "CCI_RHEUMATIC_DISEASE": 0, "CCI_PEPTIC_ULCER": 0, "CCI_MILD_LIVER_DISEASE": 0,
        "CCI_DM_NO_CC": 0, "CCI_DM_WITH_CC": 1, "CCI_HEMIPLEGIA": 0,
        "CCI_RENAL_DISEASE": 1, "CCI_MALIG_NO_SKIN": 0, "CCI_SEVERE_LIVER_DISEASE": 0,
        "CCI_METASTATIC_TUMOR": 0, "CCI_AIDS_HIV": 0, "CCI_TOTAL_SCORE": 5,
        "ELIX_CARDIAC_ARRTHYTHMIAS": 1, "ELIX_CONGESTIVE_HEART_FAILURE": 1,
        "ELIX_VALVULAR_DISEASE": 1, "ELIX_PULM_CIRC_DISORDERS": 0,
        "ELIX_PERIPH_VASC_DISEASE": 1, "ELIX_HYPERTENSION": 1, "ELIX_PARALYSIS": 0,
        "ELIX_NEURO_DISORDERS": 0, "ELIX_COPD": 1, "ELIX_DIABETES_WO_CC": 0,
        "ELIX_DIABETES_W_CC": 1, "ELIX_HYPOTHYROIDISM": 0, "ELIX_RENAL_FAILURE": 1,
        "ELIX_LIVER_DISEASE": 0, "ELIX_CHRONIC_PEPTIC_ULCER_DISEASE": 0, "ELIX_HIV_AIDS": 0,
        "ELIX_LYMPHOMA": 0, "ELIX_METASTATIC_CANCER": 0, "ELIX_TUMOR_WO_METASTATIC_CANCER": 0,
        "ELIX_RHEUMATOID_ARTHRITIS": 0, "ELIX_COAGULATION_DEFICIENCY": 0, "ELIX_OBESITY": 1,
        "ELIX_WEIGHT_LOSS": 0, "ELIX_FLUID_ELECTROLYTE_DISORDERS": 1,
        "ELIX_ANEMIA_BLOOD_LOSS": 0, "ELIX_DEFICIENCY_ANEMIAS": 1,
        "ELIX_ALCOHOL_ABUSE": 0, "ELIX_DRUG_ABUSE": 0, "ELIX_PSYCHOSES": 0, "ELIX_DEPRESSION": 1,
        "ELIX_AHRQ_SCORE": 8, "ELIX_VAN_WALRAVEN_SCORE": 12,
        "MED_CURRENT_ASA": 1, "MED_CURRENT_STATIN": 1, "MED_CURRENT_LOW_STATIN": 0,
        "MED_CURRENT_MODERATE_STATIN": 0, "MED_CURRENT_HIGH_STATIN": 1,
        "MED_CURRENT_BB": 1, "MED_CURRENT_AB": 1, "MED_CURRENT_CCB": 1,
        "MED_CURRENT_ARB": 1, "MED_CURRENT_ZETIA": 1, "MED_CURRENT_PCSK9": 0,
        "MED_CURRENT_WARFARIN": 1, "MED_CURRENT_DOAC": 0, "MED_CURRENT_COLCHICINE": 0,
        "MED_CURRENT_ARNI": 1, "MED_CURRENT_HYDRALAZINE": 0, "MED_CURRENT_MRA": 1,
        "MED_CURRENT_SPIRONOLACTONE": 1, "MED_CURRENT_MEMORY_AGENT": 0,
        "Y00_HGB_A1C": 9.2, "Y00_TRIGLYCERIDE": 280, "Y00_HDL": 32,
        "Y00_LDL": 165, "Y00_CHOLESTEROL": 245, "Y00_HSCRP": 8.5,
        "LAST_MED_ENC_TYPE": "Inpatient"
    }
    
    return low_risk, high_risk


def main():
    """Main function to wrap the secondary CVD risk model and test it."""
    output_lines = []
    
    def log(message):
        """Log a message to both console and output list."""
        print(message)
        output_lines.append(message)
    
    log("=" * 70)
    log("Wrapping Secondary CVD Risk Model for Azure ML Responsible AI Dashboard")
    log("=" * 70)
    log(f"Timestamp: {datetime.now().isoformat()}")
    log("")
    
    # Step 1: Load the original model
    log("Step 1: Loading the original secondary CVD risk model...")
    try:
        original_model = load_secondary_cvd_model()
        log(f"  ✓ Model loaded successfully")
        log(f"  ✓ Model type: {type(original_model).__name__}")
        
        # Check for model attributes
        if hasattr(original_model, 'feature_names_in_'):
            log(f"  ✓ Feature names: {len(original_model.feature_names_in_)} features")
        if hasattr(original_model, 'classes_'):
            log(f"  ✓ Classes: {original_model.classes_}")
        if hasattr(original_model, 'n_features_in_'):
            log(f"  ✓ Number of features: {original_model.n_features_in_}")
    except Exception as e:
        log(f"  ✗ Error loading model: {e}")
        return
    
    log("")
    
    # Step 2: Create the RAI wrapper
    log("Step 2: Creating RAI Model Wrapper...")
    feature_names = get_secondary_cvd_feature_names()
    
    # Determine model type based on model capabilities
    has_predict_proba = hasattr(original_model, 'predict_proba')
    model_type = 'classification' if has_predict_proba else 'regression'
    
    wrapper = ModelWrapper(
        model=original_model,
        model_type=model_type,
        feature_names=feature_names
    )
    log(f"  ✓ Wrapper created")
    log(f"  ✓ Model type: {model_type}")
    log(f"  ✓ Has predict_proba: {has_predict_proba}")
    log(f"  ✓ Number of features: {len(feature_names)}")
    log("")
    
    # Step 3: Test with sample patient data
    log("Step 3: Testing wrapper with sample patient data...")
    low_risk, high_risk = create_sample_patient_data()
    
    # Create DataFrames
    df_low = pd.DataFrame([low_risk])
    df_high = pd.DataFrame([high_risk])
    
    log("")
    log("  Testing Low Risk Patient:")
    try:
        pred_low = wrapper.predict(df_low)
        log(f"    ✓ Prediction: {pred_low}")
        
        if has_predict_proba:
            prob_low = wrapper.predict_proba(df_low)
            log(f"    ✓ Probabilities: {prob_low}")
    except Exception as e:
        log(f"    ✗ Error: {e}")
    
    log("")
    log("  Testing High Risk Patient:")
    try:
        pred_high = wrapper.predict(df_high)
        log(f"    ✓ Prediction: {pred_high}")
        
        if has_predict_proba:
            prob_high = wrapper.predict_proba(df_high)
            log(f"    ✓ Probabilities: {prob_high}")
    except Exception as e:
        log(f"    ✗ Error: {e}")
    
    log("")
    
    # Step 4: Save the wrapped model
    log("Step 4: Saving wrapped model for RAI dashboard...")
    output_dir = Path(__file__).parent / "rai_secondary_cvd_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save model
        model_path = output_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(original_model, f)
        log(f"  ✓ Model saved to: {model_path}")
        
        # Save conda environment
        conda_path = output_dir / "conda_env.yml"
        save_conda_env(str(conda_path))
        log(f"  ✓ Conda environment saved to: {conda_path}")
        
        # Save wrapper info
        wrapper_info = {
            "model_name": "secondary_cvd_risk",
            "model_type": model_type,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "has_predict_proba": has_predict_proba,
            "wrapped_date": datetime.now().isoformat()
        }
        
        import json
        info_path = output_dir / "wrapper_info.json"
        with open(info_path, 'w') as f:
            json.dump(wrapper_info, f, indent=2)
        log(f"  ✓ Wrapper info saved to: {info_path}")
        
    except Exception as e:
        log(f"  ✗ Error saving: {e}")
    
    log("")
    
    # Step 5: Generate Azure ML registration code
    log("Step 5: Azure ML Registration Code")
    log("-" * 50)
    registration_code = f'''
# Register the wrapped model in Azure ML
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Connect to workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# Register as CUSTOM_MODEL for RAI dashboard
model = Model(
    path="{output_dir}",
    type=AssetTypes.CUSTOM_MODEL,
    name="secondary-cvd-risk-rai",
    description="Secondary CVD Risk model wrapped for Responsible AI dashboard",
)

registered_model = ml_client.models.create_or_update(model)
print(f"Model registered: {{registered_model.name}}:{{registered_model.version}}")
'''
    log(registration_code)
    
    log("")
    log("=" * 70)
    log("Summary")
    log("=" * 70)
    log(f"  Original model path: models/secondary_cvd_risk/1/model.pkl")
    log(f"  Output directory: {output_dir}")
    log(f"  Model type: {model_type}")
    log(f"  Features: {len(feature_names)}")
    log(f"  Ready for RAI dashboard: Yes")
    log("=" * 70)
    
    # Save output to text file
    output_file = Path(__file__).parent / "wrap_secondary_cvd_output.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Output saved to: {output_file}")


if __name__ == "__main__":
    main()
