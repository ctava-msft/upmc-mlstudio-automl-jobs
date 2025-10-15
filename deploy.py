import os
import datetime
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_ML_WORKSPACE")

# Create ML Client
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# Define an endpoint name
endpoint_name = "endpt-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# Create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Secondary CVD Risk Endpoint",
    tags={"app": "secondary-cvd-risk"},
)

# Define environment
env = Environment(
    conda_file="./models/secondary_cvd_risk/1/conda_env_v_1_0_0.yml",
    #image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    image="mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:latest"
)

# Create blue deployment
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=Model(path="./models/secondary_cvd_risk/1/model.pkl"),
    environment=env,
    code_configuration=CodeConfiguration(
        code="./models/secondary_cvd_risk/4",
        scoring_script="scoring_wrapper.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

# Deploy (local=True for local testing)
ml_client.online_deployments.begin_create_or_update(
    deployment=blue_deployment, local=True
)
