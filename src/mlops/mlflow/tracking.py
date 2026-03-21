import mlflow
import mlflow.pytorch
from config.settings import settings

REGISTERED_MODEL_NAME = "Banking77_Classifier"
EXPERIMENT_NAME = "banking77-intent-classification"

def setup_mlflow():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_experiment(
    model=None,
    metrics: dict | None = None,
    params: dict | None = None,
    artifacts: dict | None = None,
    model_artifact_path: str = "model",
):
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)

    if model is not None:
        mlflow.pytorch.log_model(model, artifact_path=model_artifact_path)

    if artifacts:
        for name, path in artifacts.items():
            mlflow.log_artifact(path, artifact_path=name)
