import mlflow
from mlflow.artifacts import download_artifacts

mlflow.set_tracking_uri(uri="http://localhost:8000")

model_uri = "models:/nn_model_0/latest"
model = mlflow.pyfunc.load_model(model_uri)
print(model)

local_artifact_path = download_artifacts(model_uri, dst_path="./artifacts")
print(f"Artifacts descargados a: {local_artifact_path}")