import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from src.data.transformer_dataset import TransformerDataset
from src.models.transformers.transformer import BankingTransformer
from src.training.transformers.train_transformer import train_transformer
from mlflow_config.tracking import setup_mlflow
from src.evaluation.transformers.evaluate import evaluate_transformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR_ROOT = PROJECT_ROOT / "data/processed"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISTILBERT_MODEL = "distilbert-base-uncased"

setup_mlflow("Transformer_models")

train_df = pd.read_csv(DATA_DIR_ROOT / 'train_df.csv')
val_df = pd.read_csv(DATA_DIR_ROOT / 'val_df.csv')
test_df = pd.read_csv(DATA_DIR_ROOT / 'test_df.csv')

X_train, y_train = train_df['text'].tolist(), train_df['label'].tolist()
X_val, y_val = val_df['text'].tolist(), val_df['label'].tolist()
X_test, y_test = test_df['text'].tolist(), test_df['label'].tolist()

model_wrapper_distilbert = BankingTransformer(
    model_name=DISTILBERT_MODEL,
    num_labels=len(set(y_train)),
    max_length=64,
    device=DEVICE
)

print("Обучение DistilBERT модели (distilbert-base-uncased)\n")
train_transformer(model_wrapper_distilbert, X_train, y_train, X_val, y_val, epochs=10, batch_size=16, lr=2e-5)


test_ds = TransformerDataset(X_test, y_test, model_wrapper_distilbert.tokenizer, model_wrapper_distilbert.max_length)

test_dataloader = DataLoader(test_ds, batch_size=16)

test_metrics_distilbert = evaluate_transformer(model_wrapper_distilbert, test_dataloader, DEVICE)

print("Метрики DistilBERT на тестирующей выборке:\n")
print(test_metrics_distilbert['classification_report'])
