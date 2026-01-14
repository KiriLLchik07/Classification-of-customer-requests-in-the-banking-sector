import torch
from pathlib import Path
import pandas as pd

from src.models.transformers.transformer import BankingTransformer
from src.training.transformers.train_transformer import train_transformer
from mlflow_config.tracking import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR_ROOT = PROJECT_ROOT / "data/processed"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL = "bert-base-uncased"
DISTILBERT_MODEL = "distilbert-base-uncased"

setup_mlflow("Transformer_models")

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

model_wrapper_distilbert = BankingTransformer(
    model_name=DISTILBERT_MODEL,
    num_labels=len(set(y_train)),
    max_length=64,
    device=DEVICE
)

print("Обучение BERT модели (bert-base-uncased)\n")
train_transformer(model_wrapper_bert, X_train, y_train, X_val, y_val, epochs=10, batch_size=16, lr=2e-5)
