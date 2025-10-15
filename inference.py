import requests
import json
import pandas as pd
import numpy as np
import argparse
import sys
import socket
import time
from datetime import datetime

def check_port_open(host, port, timeout=2):
    """Check if the specified port is open on the host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def format_data(sample_dict):
    """Create external function format from sample input"""
    # Define the expected column order (must match model training order)
    columns = [
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
    
    # Ensure we have exactly 110 columns as expected by the model
    print(f"Expected columns count: {len(columns)}")
    
    # Create a DataFrame to ensure proper structure
    df_data = {}
    for col in columns:
        df_data[col] = [sample_dict.get(col, None)]
    
    # Create DataFrame and verify structure
    df = pd.DataFrame(df_data)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns count: {len(df.columns)}")
    
    # Convert to JSON-serializable format first
    def convert_to_json_serializable(obj):
        """Convert numpy/pandas types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, datetime, np.datetime64)):
            if pd.isna(obj):
                return None
            return pd.Timestamp(obj).isoformat()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, str):
            return obj
        elif obj is None:
            return None
        else:
            # For any other type, try to convert to native Python type
            try:
                return obj.item() if hasattr(obj, 'item') else obj
            except:
                return str(obj)
    
    # Create a proper DataFrame with all columns and their values
    # Use the sample_dict directly instead of recreating from array
    cleaned_sample = {}
    for col in columns:
        value = sample_dict.get(col, None)
        cleaned_sample[col] = convert_to_json_serializable(value)
    
    # Create DataFrame with the cleaned data
    df_final = pd.DataFrame([cleaned_sample])
    print(f"Final DataFrame shape: {df_final.shape}")
    print(f"Final DataFrame columns: {len(df_final.columns)}")
    print(f"Sample final values: {list(df_final.iloc[0].values)[:5]}...")
    
    # Instead of converting to array, send as pandas DataFrame format
    # that preserves column names for the model
    return {
        "data": df_final.to_dict('split')  # Split format preserves column names and structure
    }

def prepare_data_for_container(sample_input):
    """
    Prepare and format all data for the container, including proper date handling.
    This centralizes all data formatting in the test script.
    """
    # Define date columns that need special handling
    date_columns = ["ADMITDATE", "PROC_DATE", "DISCHARGEDATE"]
    
    # Create a copy to avoid modifying the original
    processed_input = sample_input.copy()
    
    # Convert date fields to proper pandas datetime format first
    df = pd.DataFrame([processed_input])
    
    # Convert string dates to datetime64[ns] - this ensures proper data types
    for col in date_columns:
        if col in df.columns:
            print(f"Converting {col} from {df[col].dtype} to datetime64[ns]")
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Ensure the dtype is exactly datetime64[ns]
            if df[col].dtype != 'datetime64[ns]':
                df[col] = df[col].astype('datetime64[ns]')
            print(f"  {col} is now: {df[col].dtype}")
    
    # Log data types to help with debugging
    print("\nDate column types after conversion:")
    for col in date_columns:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype} - Sample value: {df[col].iloc[0]}")
    
    # Convert the DataFrame back to a dictionary
    processed_dict = df.iloc[0].to_dict()
    
    # Convert any pandas/numpy types to JSON-serializable types
    def convert_to_json_serializable(obj):
        """Convert numpy/pandas types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, datetime, np.datetime64)):
            if pd.isna(obj):
                return None
            # Convert to ISO format string for JSON compatibility
            return pd.Timestamp(obj).isoformat()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, str):
            return obj
        elif obj is None:
            return None
        else:
            # For any other type, try to convert using .item() if available
            try:
                return obj.item() if hasattr(obj, 'item') else obj
            except:
                return str(obj)
    
    # Clean the processed dictionary
    processed_dict = convert_to_json_serializable(processed_dict)
    
    # Verify that datetime columns are properly converted
    print("\nPost-processing datetime verification:")
    for col in date_columns:
        if col in processed_dict:
            val = processed_dict[col]
            print(f"  {col}: {type(val)} = {val}")
    
    return processed_dict

def test_sample(sample_name, sample_data, docker_api_url):
    """Test a single sample and return results"""
    print(f"\n=== TESTING {sample_name.upper()} ===")
    
    # Process the data to ensure proper formatting for the container
    processed_sample = prepare_data_for_container(sample_data)
    
    # Format the data for the external function
    payload = format_data(processed_sample)
    
    try:
        # Ensure the payload is JSON serializable before sending
        try:
            json.dumps(payload)  # Test serialization
            print("✓ Payload is JSON serializable")
        except TypeError as json_error:
            print(f"⚠ JSON serialization error: {json_error}")
            return None
        
        print(f"Sending request to {docker_api_url}...")
        response = requests.post(docker_api_url, json=payload, timeout=15)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✓ SUCCESS! {sample_name} prediction result:")
                print(json.dumps(result, indent=2))
                
                # Extract and display the risk score
                if isinstance(result, dict) and 'data' in result:
                    risk_scores = result['data']
                    if risk_scores:
                        risk_score = risk_scores[0] if isinstance(risk_scores[0], (int, float)) else risk_scores[0][0]
                        print(f"📊 CVD Risk Score: {risk_score:.4f}")
                        risk_level = "HIGH RISK" if risk_score > 0.5 else "LOW RISK"
                        print(f"📈 Risk Level: {risk_level}")
                
                return result
            except json.JSONDecodeError:
                print(f"Response is not valid JSON: {response.text[:200]}...")
                return None
        else:
            print(f"Error response: {response.text[:300]}...")
            return None
            
    except Exception as e:
        print(f"{sample_name} test failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test the secondary CVD risk model in a Docker container")
    parser.add_argument("--host", default="127.0.0.1", help="Docker container host (default: 127.0.0.1)")
    parser.add_argument("--port", default="5001", help="Docker container port (default: 5001)")
    parser.add_argument("--endpoint", default="score", help="API endpoint (default: score)")
    parser.add_argument("--retry", type=int, default=3, help="Number of connection retries (default: 3)")
    args = parser.parse_args()
    
    host = args.host
    port = int(args.port)
    endpoint = args.endpoint
    max_retries = args.retry
    
    DOCKER_API_URL = f"http://{host}:{port}/{endpoint}"
    print(f"Connecting to Docker container at: {DOCKER_API_URL}")
    
    # Check if the port is open before attempting to connect
    if not check_port_open(host, port):
        print(f"\nERROR: Could not connect to {host}:{port}")
        print("Please check that:")
        print("  1. The Docker container is running")
        print("  2. The port is correct and exposed")
        print(f"  3. There are no firewall rules blocking connections to {host}:{port}")
        print("\nYou can try running:")
        print(f"  docker ps  # To check running containers")
        print(f"  docker logs <container_id>  # To check for errors in the container")
        return
    
    # Create a sample input record with all required fields - LOW CVD RISK
    neg_cvdrisk_input = {
        "AGE": 45,  # Younger age
        "SEX": 0,   # Female
        "RACE": 2,
        "RACE_LABEL": "White",
        "ETHNICITY": 1,
        "ETHNICITY_DETAILED": 1,
        "ADMITDATE": "2023-01-15",
        "PROC_DATE": "2023-01-16",
        "DISCHARGEDATE": "2023-01-20",
        "HOSPITAL": "Main",
        "PROC_TYPE": 1,
        "BMI_IP": 22.0,  # Normal BMI
        "SBP_FIRST": 120,  # Normal blood pressure
        "SBP_LAST": 118,
        "DBP_FIRST": 78,
        "DBP_LAST": 76,
        "HR_FIRST": 70,
        "HR_LAST": 68,
        "CANCER_DX": 0,
        "ALZHEIMER_DX": 0,
        "NONSPECIFIC_MCI_DX": 0,
        "VASCULAR_COGNITIVE_IMPAIRMENT_DX": 0,
        "NONSPECIFIC_COGNITIVE_DEFICIT_DX": 0,
        "CHF_HST": 0,  # No heart failure
        "DIAB_HST": 0,  # No diabetes
        "AFIB_HST": 0,
        "OBESE_HST": 0,  # Not obese
        "MORBIDOBESE_HST": 0,
        "TIA_HST": 0,
        "CARDIOMYOPATHY_HST": 0,
        "TOBACCO_STATUS": 0,  # Non-smoker
        "TOBACCO_STATUS_LABEL": "Non-smoker",
        "ALCOHOL_STATUS": 0,
        "ALCOHOL_STATUS_LABEL": "Non-drinker",
        "ILL_DRUG_STATUS": 0,
        "ILL_DRUG_STATUS_LABEL": "None",
        "CCI_CHF": 0,
        "CCI_PERIPHERAL_VASC": 0,
        "CCI_DEMENTIA": 0,
        "CCI_COPD": 0,
        "CCI_RHEUMATIC_DISEASE": 0,
        "CCI_PEPTIC_ULCER": 0,
        "CCI_MILD_LIVER_DISEASE": 0,
        "CCI_DM_NO_CC": 0,  # No diabetes
        "CCI_DM_WITH_CC": 0,
        "CCI_HEMIPLEGIA": 0,
        "CCI_RENAL_DISEASE": 0,
        "CCI_MALIG_NO_SKIN": 0,
        "CCI_SEVERE_LIVER_DISEASE": 0,
        "CCI_METASTATIC_TUMOR": 0,
        "CCI_AIDS_HIV": 0,
        "CCI_TOTAL_SCORE": 0,  # Low comorbidity score
        "ELIX_CARDIAC_ARRTHYTHMIAS": 0,
        "ELIX_CONGESTIVE_HEART_FAILURE": 0,
        "ELIX_VALVULAR_DISEASE": 0,
        "ELIX_PULM_CIRC_DISORDERS": 0,
        "ELIX_PERIPH_VASC_DISEASE": 0,
        "ELIX_HYPERTENSION": 0,  # No hypertension
        "ELIX_PARALYSIS": 0,
        "ELIX_NEURO_DISORDERS": 0,
        "ELIX_COPD": 0,
        "ELIX_DIABETES_WO_CC": 0,
        "ELIX_DIABETES_W_CC": 0,
        "ELIX_HYPOTHYROIDISM": 0,
        "ELIX_RENAL_FAILURE": 0,
        "ELIX_LIVER_DISEASE": 0,
        "ELIX_CHRONIC_PEPTIC_ULCER_DISEASE": 0,
        "ELIX_HIV_AIDS": 0,
        "ELIX_LYMPHOMA": 0,
        "ELIX_METASTATIC_CANCER": 0,
        "ELIX_TUMOR_WO_METASTATIC_CANCER": 0,
        "ELIX_RHEUMATOID_ARTHRITIS": 0,
        "ELIX_COAGULATION_DEFICIENCY": 0,
        "ELIX_OBESITY": 0,  # Not obese
        "ELIX_WEIGHT_LOSS": 0,
        "ELIX_FLUID_ELECTROLYTE_DISORDERS": 0,
        "ELIX_ANEMIA_BLOOD_LOSS": 0,
        "ELIX_DEFICIENCY_ANEMIAS": 0,
        "ELIX_ALCOHOL_ABUSE": 0,
        "ELIX_DRUG_ABUSE": 0,
        "ELIX_PSYCHOSES": 0,
        "ELIX_DEPRESSION": 0,
        "ELIX_AHRQ_SCORE": 0,  # Low score
        "ELIX_VAN_WALRAVEN_SCORE": 0,
        "MED_CURRENT_ASA": 0,  # No cardioprotective meds
        "MED_CURRENT_STATIN": 0,
        "MED_CURRENT_LOW_STATIN": 0,
        "MED_CURRENT_MODERATE_STATIN": 0,
        "MED_CURRENT_HIGH_STATIN": 0,
        "MED_CURRENT_BB": 0,
        "MED_CURRENT_AB": 0,
        "MED_CURRENT_CCB": 0,
        "MED_CURRENT_ARB": 0,
        "MED_CURRENT_ZETIA": 0,
        "MED_CURRENT_PCSK9": 0,
        "MED_CURRENT_WARFARIN": 0,
        "MED_CURRENT_DOAC": 0,
        "MED_CURRENT_COLCHICINE": 0,
        "MED_CURRENT_ARNI": 0,
        "MED_CURRENT_HYDRALAZINE": 0,
        "MED_CURRENT_MRA": 0,
        "MED_CURRENT_SPIRONOLACTONE": 0,
        "MED_CURRENT_MEMORY_AGENT": 0,
        "Y00_HGB_A1C": 5.2,  # Normal A1C
        "Y00_TRIGLYCERIDE": 90,  # Normal triglycerides
        "Y00_HDL": 60,  # Good HDL
        "Y00_LDL": 95,  # Normal LDL
        "Y00_CHOLESTEROL": 180,  # Normal cholesterol
        "Y00_HSCRP": 0.8,  # Low inflammation
        "LAST_MED_ENC_TYPE": "Outpatient"
    }

    # Create a HIGH CVD RISK sample
    pos_cvdrisk_input = {
        "AGE": 72,  # Older age
        "SEX": 1,   # Male
        "RACE": 2,
        "RACE_LABEL": "White",
        "ETHNICITY": 1,
        "ETHNICITY_DETAILED": 1,
        "ADMITDATE": "2023-01-15",
        "PROC_DATE": "2023-01-16",
        "DISCHARGEDATE": "2023-01-20",
        "HOSPITAL": "Main",
        "PROC_TYPE": 1,
        "BMI_IP": 32.5,  # Obese
        "SBP_FIRST": 165,  # High blood pressure
        "SBP_LAST": 160,
        "DBP_FIRST": 95,
        "DBP_LAST": 90,
        "HR_FIRST": 85,
        "HR_LAST": 82,
        "CANCER_DX": 0,
        "ALZHEIMER_DX": 0,
        "NONSPECIFIC_MCI_DX": 0,
        "VASCULAR_COGNITIVE_IMPAIRMENT_DX": 0,
        "NONSPECIFIC_COGNITIVE_DEFICIT_DX": 0,
        "CHF_HST": 1,  # Heart failure history
        "DIAB_HST": 1,  # Diabetes
        "AFIB_HST": 1,  # Atrial fibrillation
        "OBESE_HST": 1,
        "MORBIDOBESE_HST": 1,
        "TIA_HST": 1,  # Previous TIA
        "CARDIOMYOPATHY_HST": 1,
        "TOBACCO_STATUS": 2,  # Current smoker
        "TOBACCO_STATUS_LABEL": "Current smoker",
        "ALCOHOL_STATUS": 1,
        "ALCOHOL_STATUS_LABEL": "Social drinker",
        "ILL_DRUG_STATUS": 0,
        "ILL_DRUG_STATUS_LABEL": "None",
        "CCI_CHF": 1,
        "CCI_PERIPHERAL_VASC": 1,
        "CCI_DEMENTIA": 0,
        "CCI_COPD": 1,
        "CCI_RHEUMATIC_DISEASE": 0,
        "CCI_PEPTIC_ULCER": 0,
        "CCI_MILD_LIVER_DISEASE": 0,
        "CCI_DM_NO_CC": 0,
        "CCI_DM_WITH_CC": 1,  # Diabetes with complications
        "CCI_HEMIPLEGIA": 0,
        "CCI_RENAL_DISEASE": 1,
        "CCI_MALIG_NO_SKIN": 0,
        "CCI_SEVERE_LIVER_DISEASE": 0,
        "CCI_METASTATIC_TUMOR": 0,
        "CCI_AIDS_HIV": 0,
        "CCI_TOTAL_SCORE": 5,  # High comorbidity score
        "ELIX_CARDIAC_ARRTHYTHMIAS": 1,
        "ELIX_CONGESTIVE_HEART_FAILURE": 1,
        "ELIX_VALVULAR_DISEASE": 1,
        "ELIX_PULM_CIRC_DISORDERS": 0,
        "ELIX_PERIPH_VASC_DISEASE": 1,
        "ELIX_HYPERTENSION": 1,
        "ELIX_PARALYSIS": 0,
        "ELIX_NEURO_DISORDERS": 0,
        "ELIX_COPD": 1,
        "ELIX_DIABETES_WO_CC": 0,
        "ELIX_DIABETES_W_CC": 1,
        "ELIX_HYPOTHYROIDISM": 0,
        "ELIX_RENAL_FAILURE": 1,
        "ELIX_LIVER_DISEASE": 0,
        "ELIX_CHRONIC_PEPTIC_ULCER_DISEASE": 0,
        "ELIX_HIV_AIDS": 0,
        "ELIX_LYMPHOMA": 0,
        "ELIX_METASTATIC_CANCER": 0,
        "ELIX_TUMOR_WO_METASTATIC_CANCER": 0,
        "ELIX_RHEUMATOID_ARTHRITIS": 0,
        "ELIX_COAGULATION_DEFICIENCY": 0,
        "ELIX_OBESITY": 1,
        "ELIX_WEIGHT_LOSS": 0,
        "ELIX_FLUID_ELECTROLYTE_DISORDERS": 1,
        "ELIX_ANEMIA_BLOOD_LOSS": 0,
        "ELIX_DEFICIENCY_ANEMIAS": 1,
        "ELIX_ALCOHOL_ABUSE": 0,
        "ELIX_DRUG_ABUSE": 0,
        "ELIX_PSYCHOSES": 0,
        "ELIX_DEPRESSION": 1,
        "ELIX_AHRQ_SCORE": 8,  # High score
        "ELIX_VAN_WALRAVEN_SCORE": 12,
        "MED_CURRENT_ASA": 1,  # On multiple cardioprotective meds
        "MED_CURRENT_STATIN": 1,
        "MED_CURRENT_LOW_STATIN": 0,
        "MED_CURRENT_MODERATE_STATIN": 0,
        "MED_CURRENT_HIGH_STATIN": 1,
        "MED_CURRENT_BB": 1,
        "MED_CURRENT_AB": 1,
        "MED_CURRENT_CCB": 1,
        "MED_CURRENT_ARB": 1,
        "MED_CURRENT_ZETIA": 1,
        "MED_CURRENT_PCSK9": 0,
        "MED_CURRENT_WARFARIN": 1,
        "MED_CURRENT_DOAC": 0,
        "MED_CURRENT_COLCHICINE": 0,
        "MED_CURRENT_ARNI": 1,
        "MED_CURRENT_HYDRALAZINE": 0,
        "MED_CURRENT_MRA": 1,
        "MED_CURRENT_SPIRONOLACTONE": 1,
        "MED_CURRENT_MEMORY_AGENT": 0,
        "Y00_HGB_A1C": 9.2,  # Poor diabetes control
        "Y00_TRIGLYCERIDE": 280,  # High triglycerides
        "Y00_HDL": 32,  # Low HDL
        "Y00_LDL": 165,  # High LDL
        "Y00_CHOLESTEROL": 245,  # High cholesterol
        "Y00_HSCRP": 8.5,  # High inflammation
        "LAST_MED_ENC_TYPE": "Inpatient"
    }

    # Test both samples
    print("🏥 Testing Secondary CVD Risk Model with Two Patient Profiles")
    print("=" * 60)
    
    # Test negative CVD risk sample
    neg_result = test_sample("Negative CVD Risk Patient", neg_cvdrisk_input, DOCKER_API_URL)
    
    # Test positive CVD risk sample  
    pos_result = test_sample("Positive CVD Risk Patient", pos_cvdrisk_input, DOCKER_API_URL)
    
    # Summary comparison
    print(f"\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    if neg_result and pos_result:
        try:
            neg_score = neg_result['data'][0] if isinstance(neg_result['data'][0], (int, float)) else neg_result['data'][0][0]
            pos_score = pos_result['data'][0] if isinstance(pos_result['data'][0], (int, float)) else pos_result['data'][0][0]
            
            print(f"Low Risk Patient Score:  {neg_score:.4f}")
            print(f"High Risk Patient Score: {pos_score:.4f}")
            print(f"Score Difference:        {pos_score - neg_score:.4f}")
            
            if pos_score > neg_score:
                print("✅ Model correctly identified higher risk in the high-risk patient")
            else:
                print("⚠️  Model did not show expected risk difference")
                
        except Exception as e:
            print(f"Could not compare scores: {e}")
    else:
        if not neg_result:
            print("❌ Low risk patient test failed")
        if not pos_result:
            print("❌ High risk patient test failed")

    # ...existing code...

if __name__ == "__main__":
    main()