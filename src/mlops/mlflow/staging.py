import mlflow
from mlflow.exceptions import RestException
from config.settings import settings

def _latest_model_version(client: mlflow.MlflowClient, model_name: str) -> str | None:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))

def set_aliases_to_latest(model_name: str, aliases: list[str]) -> None:
    client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    latest_version = _latest_model_version(client, model_name)

    if latest_version is None:
        print(f"Skip: model '{model_name}' not found in registry.")
        return

    for alias in aliases:
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=latest_version,
            )
            print(f"{model_name}: alias '{alias}' -> version {latest_version}")
        except RestException as exc:
            print(f"{model_name}: failed to set alias '{alias}': {exc}")

if __name__ == "__main__":
    alias_plan = {
        "Banking77_LogisticRegression": ["baseline"],
        "Banking77_LSTM": ["reserve"],
        "Banking77_GRU": ["reserve"],
        "Banking77_BERT": ["production"],
        "Banking77_DistilBERT": ["production"],
    }

    for model_name, aliases in alias_plan.items():
        set_aliases_to_latest(model_name, aliases)
