import mlflow
import mlflow.pytorch
from pathlib import Path
import torch
from src.config.settings import settings

EXPERIMENT_NAME = 'Banking77_RNN'

def setup_mlflow():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("Banking77_RNN")

def log_rnn_experiment(model, tokenizer, metrics: dict, params: dict, run_name: str):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, name='model')

        vocab_path = Path("vocab.json")
        torch.save(tokenizer.word2idx, vocab_path)
        mlflow.log_artifact(vocab_path)

        vocab_path.unlink()
