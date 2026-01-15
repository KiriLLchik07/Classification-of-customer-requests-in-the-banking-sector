import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import mlflow

from src.data.transformer_dataset import TransformerDataset
from src.models.transformers.transformer import BankingTransformer
from src.training.transformers.train_transformer import train_transformer
from mlflow_config.tracking import setup_mlflow, log_experiment
from mlflow_config.registry import register_model, set_model_description
from mlflow_config.logging import log_environment, seed_everything, log_git_commit
from mlflow_config.dataset import file_md5
from src.evaluation.transformers.evaluate import evaluate_transformer

seed_everything(42)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR_ROOT = PROJECT_ROOT / "data/processed"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL = "bert-base-uncased"

setup_mlflow()

train_df = pd.read_csv(DATA_DIR_ROOT / 'train_df.csv')
val_df = pd.read_csv(DATA_DIR_ROOT / 'val_df.csv')
test_df = pd.read_csv(DATA_DIR_ROOT / 'test_df.csv')

X_train, y_train = train_df['text'].tolist(), train_df['label'].tolist()
X_val, y_val = val_df['text'].tolist(), val_df['label'].tolist()
X_test, y_test = test_df['text'].tolist(), test_df['label'].tolist()

model_wrapper_bert = BankingTransformer(
    model_name=BERT_MODEL,
    num_labels=len(set(y_train)),
    max_length=64,
    device=DEVICE
)

print("Обучение BERT модели (bert-base-uncased)\n")
model, val_metrics, training_time = train_transformer(model_wrapper_bert, X_train, y_train, X_val, y_val, epochs=10, batch_size=16, lr=2e-5)
best_val_metric = max(val_metrics)

test_ds = TransformerDataset(X_test, y_test, model_wrapper_bert.tokenizer, model_wrapper_bert.max_length)

test_dataloader = DataLoader(test_ds, batch_size=16)

test_metrics_bert = evaluate_transformer(model_wrapper_bert, test_dataloader, DEVICE)

print("Метрики BERT на тестирующей выборке:\n")
print(test_metrics_bert['f1_macro'])

with mlflow.start_run(run_name="BERT") as run:
    log_environment(DEVICE)

    log_git_commit()

    mlflow.log_metric("training_time_sec", training_time)

    mlflow.log_param("train_df_md5", file_md5(DATA_DIR_ROOT / 'train_df.csv'))
    mlflow.log_param("val_df_md5", file_md5(DATA_DIR_ROOT / 'val_df.csv'))
    mlflow.log_param("test_df_md5", file_md5(DATA_DIR_ROOT / 'test_df.csv'))

    log_experiment(
        model,
        metrics={
            "f1_macro_val": best_val_metric,
            "test_f1_macro": test_metrics_bert["f1_macro"]
        },
        params={
            "batch_size": 16,
            "epochs": 10,
            "lr": 2e-5
        }
    )

    register_model(
        run_id=run.info.run_id,
        artifact_path="model",
        model_name="Banking77_Classifier"
    )

    set_model_description(
        "Banking77_Classifier",
        version=8,
        description="BERT дообученный на датаесете Banking77. Показывает лучшее значение метрики f1_macro, однако скорость низкая."
    )
