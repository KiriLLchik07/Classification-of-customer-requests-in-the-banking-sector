import mlflow
from src.mlflow_model.pyfunc_model import PyFunc

def log_pyfunc_model(model_dir: str, label_mapping_path: str):
    artifacts = {
        "model": model_dir,
        "label_mapping": label_mapping_path
    }

    mlflow.pyfunc.log_model(artifact_path="model", python_model=PyFunc(), artifacts=artifacts)
