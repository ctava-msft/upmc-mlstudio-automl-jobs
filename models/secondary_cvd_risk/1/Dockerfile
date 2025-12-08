FROM mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:latest

USER dockeruser

ARG MLFLOW_MODEL_NAME=LightGBM

# Conda is already installed
ENV CONDA_ENV_DIR=/opt/miniconda/envs

# Create a new conda environment and install the same version of the server
COPY ./conda_env_v_1_0_0.yml /tmp/conda.yaml
RUN conda env create -n userenv -f /tmp/conda.yaml && \
    SERVER_VERSION=$(pip show azureml-inference-server-http | grep Version | sed -e 's/.*: //') && \
    $CONDA_ENV_DIR/userenv/bin/pip install --no-cache-dir \
        azureml-inference-server-http==$SERVER_VERSION \
        pandas numpy scikit-learn \
        "werkzeug>=2.1.0" \
        python-dateutil

# Update environment variables to default to the new conda env
ENV AZUREML_CONDA_ENVIRONMENT_PATH="$CONDA_ENV_DIR/userenv"
ENV PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH"
ENV LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"

# Provide a default model directory for local runs
ENV AZUREML_MODEL_DIR="/var/mlflow_resources"
ENV MLFLOW_MODEL_FOLDER=model

# Enable detailed logging for Snowflake debugging
ENV AZUREML_LOG_LEVEL=DEBUG
ENV PYTHONPATH="${AZUREML_MODEL_DIR}:${PYTHONPATH}"

# Copy the model.pkl file to the container
COPY ./model.pkl /tmp/model.pkl

# Copy the scoring script to the model directory
# COPY ./mlflow_score_script.py ${AZUREML_MODEL_DIR}/mlflow_score_script.py

# Create a real MLflow model using the loaded model from model.pkl
USER root
RUN mkdir -p ${AZUREML_MODEL_DIR} && \
    /opt/miniconda/envs/userenv/bin/python - <<'PY'
import os
import pickle
import mlflow
import pandas as pd
import numpy as np

# Load the model from the pkl file
with open('/tmp/model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Create a PyFunc wrapper for the loaded model that handles input correctly
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
       
    def predict(self, ctx, model_input):
        try:
            print(f"DEBUG: Received model_input type: {type(model_input)}")
            print(f"DEBUG: Model input keys: {list(model_input.keys()) if isinstance(model_input, dict) else 'not a dict'}")
        
            # Make Dataframe
            df = pd.DataFrame(model_input)
           
            print(f"DEBUG: Final DataFrame shape: {df.shape}")
            print(f"DEBUG: DataFrame columns: {list(df.columns)[:5]}...")
            
            # Make prediction
            result = self.model.predict(df)
            print(f"DEBUG: Prediction result type: {type(result)}, shape: {getattr(result, 'shape', 'no shape')}")
            
            # Convert to list format
            if hasattr(result, 'tolist'):
                result_list = result.tolist()
            else:
                result_list = result if isinstance(result, list) else [result]
            
            return result_list

        except Exception as e:
            print(f"ERROR during prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

# Save the model to the MLflow model directory
dst = os.path.join(os.environ["AZUREML_MODEL_DIR"], os.environ["MLFLOW_MODEL_FOLDER"])
mlflow.pyfunc.save_model(path=dst, python_model=ModelWrapper(loaded_model), conda_env=None)
PY

RUN chown -R dockeruser:dockeruser ${AZUREML_MODEL_DIR}
USER dockeruser

EXPOSE 5001
CMD ["runsvdir", "/var/runit"]