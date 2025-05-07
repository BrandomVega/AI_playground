import mlflow
from mlflow.models import infer_signature
from mlflow.artifacts import download_artifacts, list_artifacts
import mlflow.pytorch
import torch
from torch import nn
import pandas as pd
import os
import tempfile

mlflow.set_tracking_uri(uri="http://localhost:8000")
mlflow.set_experiment("model_training")
PORT = 8001

class NNModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_uris):
        self.model_uris = model_uris
        self.models = {}
        self.artifacts = {}
        self.pathArtifacts = {}

    @staticmethod
    def _model_uri_to_idx(model_uri: str) -> int:
        return int(model_uri.split("/")[-2].split("_")[-1])

    def load_context(self, context):
        self.models = {}
        for model_uri in self.model_uris:
            model_key = self._model_uri_to_idx(model_uri)
            model = mlflow.pytorch.load_model(model_uri)
            self.artifacts[model_key] = list_artifacts(model_uri)
            tmp_dir = tempfile.mkdtemp(dir="/tmp")
            local_artifacts = download_artifacts(artifact_uri=model_uri, dst_path=tmp_dir)
            jpg_files = [f"{local_artifacts}{file}" for file in os.listdir(local_artifacts) if file.endswith('.jpg')]
            self.pathArtifacts[model_key] = jpg_files
            self.models[model_key] = model
            print(jpg_files)

    def predict(self, context, model_input, params):
        idx_model = params.get("model")
        if idx_model is None:
            raise ValueError("Model param is not passed.")

        model = self.models.get(idx_model)
        if model is None:
            raise ValueError(f"Model {idx_model} version was not found: {self.models.keys()}.")

        if isinstance(model_input, pd.DataFrame):
            model_input = torch.tensor(model_input.values, dtype=torch.float32)

        prediction = model(model_input)
        prediction_list = prediction.detach().numpy().tolist()

        return {
            "prediction": prediction_list,
            "artifacts_path": self.pathArtifacts[idx_model]
        }

NN_MODEL_NAME_PREFIX = "nn_model_"
MME_MODEL_NAME = "MME_nn_model"
NUM_MODELS = 5
model_uris = [f"models:/{NN_MODEL_NAME_PREFIX}{i}/latest" for i in range(NUM_MODELS)]
print(model_uris)

X_df = pd.DataFrame({
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1]
})
Y_df = pd.DataFrame({
    "label": [0, 0, 0, 1]
})
params = {"model": 0}
with mlflow.start_run():
    model = NNModel(model_uris)
    model.load_context(None)
    model_path = "MME_model_path"
    signature = infer_signature(model_input=X_df, model_output=Y_df, params=params)
    mlflow.pyfunc.log_model(
        model_path,
        python_model=model,
        signature=signature,
        registered_model_name=MME_MODEL_NAME
    )
    mlflow.log_param("num_models_inside", len(model.models))
print(
    f"""mlflow models serve -m "models:/MME_nn_model/latest" --env-manager local -p 8001
    mlflow models build-docker --model-uri "models:/MME_nn_model/latest" --name "modelom"
    """   
)
