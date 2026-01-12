import mlflow

def register_model(run_id: str, artifact_path: str, model_name: str):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.register_model(model_uri=model_uri, name=model_name)
