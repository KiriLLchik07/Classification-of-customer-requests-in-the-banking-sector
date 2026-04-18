import mlflow
from mlflow.exceptions import RestException
from config.settings import settings

def set_model_stages_and_aliases(model_name: str, version_info: dict):
    """
    Устанавливает aliases для версий модели в MLflow Model Registry.

    Args:
        model_name (str): Имя модели в Model Registry.
        version_info (dict): Словарь {version: [aliases]}, например:
            {
                1: ["baseline"],
                2: ["reserve"],
                3: ["production"]
            }
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    
    for version, aliases in version_info.items():
        for alias in aliases:
            try:
                client.set_registered_model_alias(
                    name=model_name,
                    alias=alias,
                    version=version
                )
                print(f"Версия {version} -> alias: {alias}")
            except RestException as e:
                print(f"Ошибка при добавлении alias '{alias}' для версии {version}: {e}")

if __name__ == "__main__":
    REGISTERED_BASELINE_NAME = "Banking77_LogisticRegression"
    REGISTERED_MODEL_NAME_LSTM = "Banking77_LSTM"
    REGISTERED_MODEL_NAME_GRU = "Banking77_GRU"
    REGISTERED_MODEL_NAME_BERT = "Banking77_BERT"
    REGISTERED_MODEL_NAME_DISTILBERT = "Banking77_DistilBERT"
    
    set_model_stages_and_aliases(REGISTERED_BASELINE_NAME, {2: ["baseline"]})
    set_model_stages_and_aliases(REGISTERED_MODEL_NAME_LSTM, {2: ["reserve"]})
    set_model_stages_and_aliases(REGISTERED_MODEL_NAME_GRU, {2: ["reserve"]})
    set_model_stages_and_aliases(REGISTERED_MODEL_NAME_BERT, {2: ["production"]})
    set_model_stages_and_aliases(REGISTERED_MODEL_NAME_DISTILBERT, {2: ["production"]})
