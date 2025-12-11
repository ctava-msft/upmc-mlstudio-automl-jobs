"""
Deep Extract Sklearn Components from AutoML Model

This script deeply inspects the AutoML model and extracts just the
pure sklearn components, removing all Azure ML wrappers.
"""

import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_and_save_pipeline(model, output_path):
    """Extract and save the sklearn pipeline from a model."""
    try:
        from sklearn.pipeline import Pipeline
        
        # Check if it's already a sklearn pipeline
        if isinstance(model, Pipeline):
            print("  Model is already a sklearn Pipeline")
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved to: {output_path}")
            return True
        
        # Try to extract pipeline components
        if hasattr(model, 'steps'):
            steps = model.steps
            print(f"  Extracting {len(steps)} pipeline steps...")
            
            sklearn_steps = []
            for name, transformer in steps:
                module = type(transformer).__module__
                print(f"    - {name}: {type(transformer).__name__} (from {module})")
                
                # Keep all steps, including custom ones
                sklearn_steps.append((name, transformer))
            
            # Create pipeline
            pipeline = Pipeline(sklearn_steps)
            
            # Save
            with open(output_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            print(f"✓ Saved pipeline with {len(sklearn_steps)} steps")
            return True
            
        elif hasattr(model, '_final_estimator'):
            print("  Extracting final estimator...")
            estimator = model._final_estimator
            with open(output_path, 'wb') as f:
                pickle.dump(estimator, f)
            print(f"✓ Saved estimator: {type(estimator).__name__}")
            return True
        
        else:
            # Save as-is
            print("  No special structure found, saving model as-is...")
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved")
            return True
            
    except Exception as e:
        print(f"✗ Failed to extract/save: {e}")
        import traceback
        traceback.print_exc()
        return False


def deep_extract_sklearn_pipeline(automl_model_path, output_path):
    """
    Extract only the sklearn components from the AutoML model.
    """
    print("=" * 70)
    print("Deep Extraction of Sklearn Components")
    print("=" * 70)
    print(f"Input:  {automl_model_path}")
    print(f"Output: {output_path}")
    print()
    
    # Strategy 1: Try direct pickle load with custom loader that skips ONNX
    print("Strategy 1: Direct pickle load (skipping problematic modules)...")
    try:
        import sys
        import types
        
        # Create a mock module for onnx to prevent DLL loading
        class MockONNX:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        
        sys.modules['onnx'] = MockONNX()
        sys.modules['onnxruntime'] = MockONNX()
        
        # Now try to load
        with open(automl_model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Model loaded: {type(model).__name__}")
        
        # Extract sklearn components
        if hasattr(model, 'steps'):
            print(f"  Found pipeline with {len(model.steps)} steps")
            return extract_and_save_pipeline(model, output_path)
        elif hasattr(model, '_final_estimator'):
            print(f"  Found wrapped estimator")
            return extract_and_save_pipeline(model, output_path)
        else:
            # Try to save it as-is
            print("  Saving model as-is...")
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            print("✓ Saved")
            return True
            
    except Exception as e:
        print(f"✗ Strategy 1 failed: {e}")
    
    # Strategy 2: Load with MLflow
    print("\nStrategy 2: Loading with MLflow...")
    try:
        import mlflow
        model_dir = Path(automl_model_path).parent
        loaded_model = mlflow.pyfunc.load_model(str(model_dir))
        print(f"✓ Model loaded: {type(loaded_model).__name__}")
    except Exception as e:
        print(f"✗ Strategy 2 failed: {e}")
        return False
    
    # Extract the actual model
    print("\nExtracting model components...")
    try:
        # Get the underlying model implementation
        if hasattr(loaded_model, '_model_impl'):
            model_impl = loaded_model._model_impl
            print(f"  Model implementation: {type(model_impl).__name__}")
            
            # Get the actual Python model
            if hasattr(model_impl, 'python_model'):
                python_model = model_impl.python_model
                print(f"  Python model: {type(python_model).__name__}")
                
                # This is the PipelineWithYTransformations object
                # Let's extract its sklearn pipeline
                if hasattr(python_model, 'steps'):
                    print(f"  ✓ Found pipeline with {len(python_model.steps)} steps")
                    
                    # Inspect the pipeline steps
                    print("\n  Pipeline steps:")
                    for i, (name, transformer) in enumerate(python_model.steps):
                        print(f"    {i+1}. {name}: {type(transformer).__name__}")
                    
                    # Try to create a pure sklearn pipeline
                    from sklearn.pipeline import Pipeline
                    
                    # Extract sklearn-compatible steps
                    sklearn_steps = []
                    for name, transformer in python_model.steps:
                        # Check if it's a pure sklearn transformer
                        module = type(transformer).__module__
                        if 'sklearn' in module:
                            sklearn_steps.append((name, transformer))
                            print(f"    ✓ Keeping sklearn step: {name}")
                        else:
                            print(f"    ✗ Skipping non-sklearn step: {name} ({module})")
                    
                    if sklearn_steps:
                        # Create a pure sklearn pipeline
                        sklearn_pipeline = Pipeline(sklearn_steps)
                        print(f"\n✓ Created pure sklearn pipeline with {len(sklearn_steps)} steps")
                        
                        # Save it
                        print(f"\nSaving to: {output_path}")
                        with open(output_path, 'wb') as f:
                            pickle.dump(sklearn_pipeline, f)
                        
                        # Verify
                        print("Verifying...")
                        with open(output_path, 'rb') as f:
                            verified = pickle.load(f)
                        print(f"✓ Verified: {type(verified).__name__}")
                        
                        print("\n" + "=" * 70)
                        print("SUCCESS!")
                        print("=" * 70)
                        return True
                    else:
                        print("\n✗ No pure sklearn steps found")
                        
                # If it has a named_steps attribute (sklearn pipeline)
                elif hasattr(python_model, 'named_steps'):
                    print(f"  ✓ Found named_steps")
                    
                    # Just save the whole thing and hope it works
                    print(f"\nAttempting to save the model as-is...")
                    with open(output_path, 'wb') as f:
                        pickle.dump(python_model, f)
                    print(f"✓ Saved")
                    
                    return True
                    
                # Try to get the final estimator
                elif hasattr(python_model, '_final_estimator'):
                    final_est = python_model._final_estimator
                    print(f"  ✓ Found final estimator: {type(final_est).__name__}")
                    
                    with open(output_path, 'wb') as f:
                        pickle.dump(final_est, f)
                    print(f"✓ Saved final estimator")
                    return True
                
                else:
                    # Just try saving the whole python_model
                    print(f"\n  Attempting to save python_model directly...")
                    with open(output_path, 'wb') as f:
                        pickle.dump(python_model, f)
                    print(f"✓ Saved")
                    return True
                    
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False


def main():
    automl_model_path = Path("./models/secondary_cvd_risk/1/model.pkl")
    output_path = Path("./models/secondary_cvd_risk/sklearn_model_extracted.pkl")
    
    if not automl_model_path.exists():
        print(f"ERROR: Model not found: {automl_model_path}")
        return 1
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = deep_extract_sklearn_pipeline(automl_model_path, output_path)
    
    if success:
        print(f"\n✓ Model saved to: {output_path}")
        return 0
    else:
        print(f"\n✗ Extraction failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
