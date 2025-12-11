"""
Extract Sklearn Estimator from AutoML Model

This script extracts the core sklearn estimator from an Azure AutoML model,
bypassing Azure ML dependencies and ONNX runtime issues.

The extracted model can be used for inference and SHAP analysis without
requiring azureml-train-automl-runtime or ONNX dependencies.
"""

import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_sklearn_model_from_automl(automl_model_path, output_path):
    """
    Extract the sklearn estimator from an AutoML model pickle file.
    
    Args:
        automl_model_path: Path to the AutoML model.pkl file
        output_path: Path where the extracted sklearn model should be saved
    
    Returns:
        True if successful, False otherwise
    """
    print("=" * 70)
    print("Extracting Sklearn Model from AutoML Model")
    print("=" * 70)
    print(f"Input:  {automl_model_path}")
    print(f"Output: {output_path}")
    print()
    
    # Try multiple extraction strategies
    strategies = [
        ("Direct unpickling with restricted globals", extract_with_restricted_globals),
        ("MLflow pyfunc unwrapping", extract_via_mlflow_pyfunc),
        ("Pickle bytes inspection", extract_via_pickle_inspection),
    ]
    
    for strategy_name, strategy_func in strategies:
        print(f"Attempting strategy: {strategy_name}...")
        try:
            model = strategy_func(automl_model_path)
            if model is not None:
                print(f"✓ Success! Extracted model type: {type(model).__name__}")
                
                # Save the extracted model
                print(f"\nSaving extracted model to: {output_path}")
                with open(output_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Verify the saved model
                print("Verifying saved model...")
                with open(output_path, 'rb') as f:
                    verified_model = pickle.load(f)
                print(f"✓ Model verified: {type(verified_model).__name__}")
                
                # Check if model has predict method
                if hasattr(verified_model, 'predict'):
                    print(f"✓ Model has predict() method")
                if hasattr(verified_model, 'predict_proba'):
                    print(f"✓ Model has predict_proba() method")
                
                print("\n" + "=" * 70)
                print("SUCCESS: Sklearn model extracted and saved!")
                print("=" * 70)
                return True
                
        except Exception as e:
            print(f"✗ Strategy failed: {e}")
            print()
    
    print("=" * 70)
    print("ERROR: All extraction strategies failed")
    print("=" * 70)
    return False


def extract_with_restricted_globals(model_path):
    """
    Try to load the model with restricted globals to bypass azureml dependencies.
    """
    # Create a custom unpickler that maps azureml classes to None or sklearn equivalents
    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Allow sklearn modules
            if module.startswith('sklearn'):
                return super().find_class(module, name)
            
            # Allow numpy, pandas
            if module in ['numpy', 'pandas', 'numpy.core.multiarray', 'numpy.core.numeric']:
                return super().find_class(module, name)
            
            # Block azureml modules and return a dummy class
            if 'azureml' in module:
                print(f"  Blocked: {module}.{name}")
                # Return None or a dummy object
                return type('Dummy', (), {})
            
            # Allow everything else
            return super().find_class(module, name)
    
    with open(model_path, 'rb') as f:
        unpickler = RestrictedUnpickler(f)
        obj = unpickler.load()
        
        # Try to extract sklearn model from the object
        if hasattr(obj, 'steps'):  # Pipeline
            return obj
        elif hasattr(obj, 'predict'):  # Direct estimator
            return obj
        elif hasattr(obj, '_final_estimator'):  # Wrapped estimator
            return obj._final_estimator
        else:
            # Inspect object attributes
            for attr in dir(obj):
                if not attr.startswith('_'):
                    val = getattr(obj, attr, None)
                    if val is not None and hasattr(val, 'predict'):
                        print(f"  Found model in attribute: {attr}")
                        return val
    
    return None


def extract_via_mlflow_pyfunc(model_path):
    """
    Try to load via MLflow and extract the underlying sklearn model.
    """
    try:
        import mlflow
        model_dir = Path(model_path).parent
        
        # Try loading as pyfunc
        loaded = mlflow.pyfunc.load_model(str(model_dir))
        
        # Try to get the underlying sklearn model
        if hasattr(loaded, '_model_impl'):
            impl = loaded._model_impl
            if hasattr(impl, 'sklearn_model'):
                return impl.sklearn_model
            elif hasattr(impl, 'python_model'):
                return impl.python_model
            elif hasattr(impl, '_model'):
                return impl._model
        
        return None
    except Exception as e:
        raise Exception(f"MLflow extraction failed: {e}")


def extract_via_pickle_inspection(model_path):
    """
    Inspect the pickle file and try to extract sklearn components.
    """
    import pickletools
    import io
    
    # Read the pickle and look for sklearn objects
    with open(model_path, 'rb') as f:
        data = f.read()
    
    # Try to find sklearn model patterns in the pickle
    if b'sklearn' in data:
        print("  Found sklearn references in pickle")
        
        # Try partial unpickling with error handling
        try:
            with open(model_path, 'rb') as f:
                # Read pickle opcodes
                opcodes = list(pickletools.genops(f))
                
                # Look for sklearn class instances
                for opcode, arg, pos in opcodes:
                    if opcode.name == 'GLOBAL' and arg and 'sklearn' in str(arg):
                        print(f"    Found sklearn class: {arg}")
        except Exception as e:
            print(f"  Pickle inspection error: {e}")
    
    return None


def main():
    """Main execution"""
    # Paths
    automl_model_path = Path("./models/secondary_cvd_risk/1/model.pkl")
    output_path = Path("./models/secondary_cvd_risk/sklearn_model_extracted.pkl")
    
    # Check if input exists
    if not automl_model_path.exists():
        print(f"ERROR: AutoML model not found at {automl_model_path}")
        return 1
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract the model
    success = extract_sklearn_model_from_automl(automl_model_path, output_path)
    
    if success:
        print(f"\n✓ Extracted model saved to: {output_path}")
        print(f"\nYou can now use this model with:")
        print(f"  import pickle")
        print(f"  with open('{output_path}', 'rb') as f:")
        print(f"      model = pickle.load(f)")
        print(f"  predictions = model.predict(X)")
        return 0
    else:
        print(f"\n✗ Failed to extract model")
        print(f"\nAlternative approaches:")
        print(f"  1. Run this in Azure ML Compute where AutoML models are natively supported")
        print(f"  2. Use the sklearn_rai_model which is already a pure sklearn model")
        print(f"  3. Retrain the model using pure sklearn (see train_sklearn_model.py)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
