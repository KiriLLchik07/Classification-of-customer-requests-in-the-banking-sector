import mlflow
import mlflow.pytorch
from pathlib import Path
import torch
from src.config.settings import settings

REGISTERED_MODEL_NAME = 'Banking77_Classifier'

def setup_mlflow(set_experiment_name: str):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(set_experiment_name)

def log_rnn_experiment(model, tokenizer, metrics: dict, params: dict, run_name: str, model_name: str):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, name=model_name, registered_model_name=REGISTERED_MODEL_NAME)

        vocab_path = Path("vocab.pt")
        torch.save(tokenizer.word2idx, vocab_path)
        mlflow.log_artifact(vocab_path)

        vocab_path.unlink()
