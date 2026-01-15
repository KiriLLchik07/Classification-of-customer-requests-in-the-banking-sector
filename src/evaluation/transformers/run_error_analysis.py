import mlflow.pytorch
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data.transformer_dataset import TransformerDataset
from src.evaluation.transformers.error_analyzer import error_analysis
from src.config.settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR_ROOT = PROJECT_ROOT / "data/processed"
REPORT_PATH = PROJECT_ROOT / "reports"

model = mlflow.pytorch.load_model(
    "models:/Banking77_Classifier/8"
)
model.to(DEVICE)

test_df = pd.read_csv(DATA_DIR_ROOT / "test_df.csv")

X_test = test_df["text"].tolist()
y_test = test_df["label"].tolist()

label2intent = (
    test_df[["label", "intent"]]
    .drop_duplicates()
    .set_index("label")["intent"]
    .to_dict()
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

test_ds = TransformerDataset(X_test, y_test, tokenizer, max_length=64)
test_dl = DataLoader(test_ds, batch_size=16)

errors_df = error_analysis(
    model=model,
    dataloader=test_dl,
    texts=X_test,
    label2intent=label2intent,
    device=DEVICE,
)

errors_df.to_csv(REPORT_PATH / "distilbert_error_analysis.csv", index=False)

confusion_matrix_ = confusion_matrix(errors_df['true_intent'], errors_df['pred_intent'], labels=sorted(label2intent.values()))

plt.figure(figsize=(16,14))
sb.heatmap(confusion_matrix_, xticklabels=sorted(label2intent.values()), 
           yticklabels=sorted(label2intent.values()), cmap='Blues', annot=True)

plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.title("Confusion Matrix - DistilBERT")
plt.show()

error_counts = (errors_df.groupby("true_intent").size().head(10))
error_counts.plot(kind="barh", figsize=(10,8), title="Классы, которые дают больше всего ошибок")
plt.xlabel("Количество ошибок")
plt.show()
