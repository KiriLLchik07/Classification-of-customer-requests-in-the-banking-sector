import mlflow

def register_model(run_id: str, artifact_path: str, model_name: str):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.register_model(model_uri=model_uri, name=model_name)

def set_model_description(model_name, version, description):
    client = mlflow.tracking.MlflowClient()
    client.update_model_version(
        name=model_name,
        version=version,
        description=description
    )
