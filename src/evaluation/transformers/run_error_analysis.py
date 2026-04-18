from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sb
from sklearn.metrics import confusion_matrix

from config.settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR_ROOT = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports"
REPORT_PATH.mkdir(parents=True, exist_ok=True)

model = mlflow.pyfunc.load_model("models:/Banking77_DistilBERT@production")

test_df = pd.read_csv(DATA_DIR_ROOT / "test_df.csv")
label_column = "intent" if "intent" in test_df.columns else "label_name"
if label_column not in test_df.columns:
    raise ValueError("Expected either 'intent' or 'label_name' column in test_df.csv")

pred_df = model.predict(test_df[["text"]])
if "prediction" not in pred_df.columns:
    raise ValueError("PyFunc model must return DataFrame with 'prediction' column")

errors_df = pd.DataFrame(
    {
        "text": test_df["text"],
        "true_intent": test_df[label_column],
        "pred_intent": pred_df["prediction"],
    }
)
errors_df = errors_df[errors_df["true_intent"] != errors_df["pred_intent"]]
errors_df.to_csv(REPORT_PATH / "distilbert_error_analysis.csv", index=False)

labels = sorted(set(test_df[label_column].astype(str).tolist()))
confusion_matrix_ = confusion_matrix(
    test_df[label_column].astype(str),
    pred_df["prediction"].astype(str),
    labels=labels,
)

plt.figure(figsize=(16, 14))
sb.heatmap(
    confusion_matrix_,
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues",
    annot=False,
)
plt.xlabel("Predicted classes")
plt.ylabel("True classes")
plt.title("Confusion Matrix - DistilBERT")
plt.show()

error_counts = errors_df.groupby("true_intent").size().head(10)
error_counts.plot(kind="barh", figsize=(10, 8), title="Top intents with most errors")
plt.xlabel("Error count")
plt.show()
