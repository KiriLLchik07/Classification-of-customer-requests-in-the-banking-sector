import mlflow
from mlflow.exceptions import RestException
from src.config.settings import settings

def set_model_stages_and_aliases(model_name: str, version_info: dict):
    """
    Устанавливает stage и aliases для версий модели в MLflow Model Registry.

    Args:
        model_name (str): Имя модели в Model Registry.
        version_info (dict): Словарь {version: (stage, [aliases])}, например:
            {
                1: ("Production", ["Best_RNN"]),
                2: ("Staging", ["Test_RNN"]),
                3: ("Archived", [])
            }
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    
    for version, (stage, aliases) in version_info.items():
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production")
            )
            print(f"Версия {version} -> stage: {stage}")
            
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
        except RestException as e:
            print(f"Ошибка при установке stage для версии {version}: {e}")


if __name__ == "__main__":
    MODEL_NAME = "Banking77_Classifier"
    
    version_dict = {
        1: ("Archived", ["Classic_ML"]), # Logistic Regression
        2: ("Archived", ["Legacy_RNN_LSTM"]), # LSTM
        3: ("Archived", ["Legacy_RNN_GRU"]), # GRU
        7: ("Staging", ["BERT_best_metrics"]), # BERT
        8: ("Production", ["Best_model", "Fast_Transformer"]) # DistilBERT
    }
    
    set_model_stages_and_aliases(MODEL_NAME, version_dict)
