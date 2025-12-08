import os
import sys

def init():
    """
    This function is called when the container is initialized/started.
    """
    global model
    
    # Get the path to where models are stored in Azure ML
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    
    # Azure ML stores models in numbered subdirectories (e.g., 1/, 2/, etc.)
    # Find the actual model.pkl file
    model_path = None
    if model_dir:
        for root, dirs, files in os.walk(model_dir):
            if 'model.pkl' in files:
                model_path = os.path.join(root, 'model.pkl')
                break
    
    if not model_path:
        raise Exception(f"Could not find model.pkl in {model_dir}")
    
    # Load the model
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    """
    import json
    import pandas as pd
    
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Return predictions as JSON
        return json.dumps(predictions.tolist())
    except Exception as e:
        return json.dumps({"error": str(e)})
