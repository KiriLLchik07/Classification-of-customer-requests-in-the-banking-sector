import mlflow
import mlflow.pytorch
from src.config.settings import settings

REGISTERED_MODEL_NAME = 'Banking77_Classifier'

def setup_mlflow(set_experiment_name: str):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(set_experiment_name)

def log_experiment(model, metrics: dict, params: dict, artifacts: dict | None = None):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.pytorch.log_model(model, artifact_path="model")

    if artifacts:
        for name, path in artifacts.items():
            mlflow.log_artifact(path, artifact_path=name)
