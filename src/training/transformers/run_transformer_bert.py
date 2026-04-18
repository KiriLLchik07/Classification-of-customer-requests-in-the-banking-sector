import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import mlflow
import json

from src.data.transformer_dataset import TransformerDataset
from src.models.transformers.transformer import BankingTransformer
from src.training.transformers.train_transformer import train_transformer
from src.mlops.mlflow.tracking import setup_mlflow, log_experiment
from src.mlops.mlflow.registry import register_model, set_model_description
from src.mlops.mlflow.logging import log_environment, seed_everything, log_git_commit
from src.mlops.mlflow.dataset import file_md5
from src.evaluation.transformers.evaluate import evaluate_transformer
from src.mlops.packaging.log_pyfunc_model import log_pyfunc_model, TransformerPyFunc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR_ROOT = PROJECT_ROOT / "data/processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL = "bert-base-uncased"
REGISTERED_MODEL_NAME_BERT = "Banking77_BERT"

def build_label_mapping(train_df: pd.DataFrame) -> dict[int, str]:
    if "label_name" in train_df.columns:
        deduplicated = train_df[["label", "label_name"]].drop_duplicates(subset=["label"])
        return {
            int(row["label"]): str(row["label_name"])
            for _, row in deduplicated.iterrows()
        }
    labels = sorted(train_df["label"].unique().tolist())
    return {int(label): str(label) for label in labels}

def run() -> None:    
    setup_mlflow()
    seed_everything(42)

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

    with mlflow.start_run(run_name="BERT") as run_info:
        log_environment(DEVICE)

        log_git_commit()

        mlflow.log_metric("training_time_sec", training_time)

        mlflow.log_param("train_df_md5", file_md5(DATA_DIR_ROOT / 'train_df.csv'))
        mlflow.log_param("val_df_md5", file_md5(DATA_DIR_ROOT / 'val_df.csv'))
        mlflow.log_param("test_df_md5", file_md5(DATA_DIR_ROOT / 'test_df.csv'))

        artifact_dir = PROJECT_ROOT / "artifacts" / "local_models" / "transformers" / "bert"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_wrapper_bert.model.save_pretrained(artifact_dir)
        model_wrapper_bert.tokenizer.save_pretrained(artifact_dir)

        label_mapping = build_label_mapping(train_df)
        with open(artifact_dir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)

        log_pyfunc_model(
            model_dir=str(artifact_dir),
            python_model=TransformerPyFunc(),
            artifact_path="model",
            pip_requirements=["mlflow", "pandas", "torch", "transformers"],
        )

        log_experiment(
            model=None,
            metrics={
                "f1_macro_val": best_val_metric,
                "test_f1_macro": test_metrics_bert["f1_macro"]
            },
            params={
                "batch_size": 16,
                "epochs": 10,
                "lr": 2e-5
            },
        )

        version = register_model(
            run_id=run_info.info.run_id,
            artifact_path="model",
            model_name=REGISTERED_MODEL_NAME_BERT
        )

        set_model_description(
            REGISTERED_MODEL_NAME_BERT,
            version=version,
            description="BERT fine-tuned on Banking77 dataset. Demonstrate excellent by f1_macro, but have slow inference speed"
        )

if __name__ == "__main__":
    run()
