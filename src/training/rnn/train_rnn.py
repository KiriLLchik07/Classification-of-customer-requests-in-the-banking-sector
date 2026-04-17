import json
from pathlib import Path
import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BankingDataset
from src.evaluation.rnn.evaluate_rnn import evaluate
from src.features.collate import collate_fn
from src.features.tokenizer import TextTokenizer
from src.mlops.mlflow.logging import log_git_commit, seed_everything
from src.mlops.mlflow.registry import register_model, set_model_description
from src.mlops.mlflow.tracking import log_experiment, setup_mlflow
from src.mlops.packaging.log_pyfunc_model import RNNPyFunc, log_pyfunc_model
from src.models.rnn.lstm_gru import RNNClassifier
from src.training.rnn.train_one_epoch import train_one_epoch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_SAVE_ROOT = PROJECT_ROOT / "artifacts" / "local_models" / "rnn"
MODEL_SAVE_ROOT.mkdir(parents=True, exist_ok=True)

REGISTERED_MODEL_NAME_LSTM = "Banking77_LSTM"
REGISTERED_MODEL_NAME_GRU = "Banking77_GRU"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 25
MAX_LEN = 80
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LR = 1e-3

def train_eval_rnn_model(model, train_dataloader, val_dataloader, test_dataloader, optimizer, criterion, device):
    last_test_metrics = {}
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        metrics_val = evaluate(model, val_dataloader, device)
        print(
            f"Epoch: {epoch + 1}"
            f"Train loss: {train_loss:.4f} | "
            f"F1-macro (val): {metrics_val['f1_macro']:.4f}"
        )
        last_test_metrics = evaluate(model, test_dataloader, device)
    return last_test_metrics

def build_label_mapping(train_df: pd.DataFrame) -> dict[int, str]:
    labels = sorted(train_df["label"].unique().tolist())
    return {int(label): str(label) for label in labels}

def save_rnn_artifacts(
    model: RNNClassifier,
    tokenizer: TextTokenizer,
    model_name: str,
    rnn_type: str,
    num_classes: int,
    label_mapping: dict[int, str],
) -> Path:
    artifact_dir = MODEL_SAVE_ROOT / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": tokenizer.word2idx,
        },
        artifact_dir / "checkpoint.pt",
    )

    config = {
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": num_classes,
        "rnn_type": rnn_type,
        "max_len": MAX_LEN,
    }
    with open(artifact_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(artifact_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    return artifact_dir

def run() -> None:
    seed_everything(42)
    setup_mlflow()

    train_df = pd.read_csv(DATA_DIR_ROOT / "train_df.csv")
    val_df = pd.read_csv(DATA_DIR_ROOT / "val_df.csv")
    test_df = pd.read_csv(DATA_DIR_ROOT / "test_df.csv")

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    label_mapping = build_label_mapping(train_df)
    num_classes = len(set(y_train))

    tokenizer = TextTokenizer()
    tokenizer.fit(X_train)

    train_dataset = BankingDataset(texts=X_train, labels=y_train, tokenizer=tokenizer, max_length=MAX_LEN)
    val_dataset = BankingDataset(texts=X_val, labels=y_val, tokenizer=tokenizer, max_length=MAX_LEN)
    test_dataset = BankingDataset(texts=X_test, labels=y_test, tokenizer=tokenizer, max_length=MAX_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model_lstm = RNNClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        rnn_type="lstm",
    ).to(DEVICE)

    model_gru = RNNClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        rnn_type="gru",
    ).to(DEVICE)

    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=LR)
    optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="LSTM") as run_info:
        log_git_commit()
        metrics_lstm = train_eval_rnn_model(
            model_lstm,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            optimizer_lstm,
            criterion,
            DEVICE,
        )
        params_lstm = {
            "model_type": "LSTM",
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "device": str(DEVICE),
        }
        log_experiment(
            model=None,
            metrics={"test_f1_macro": metrics_lstm["f1_macro"]},
            params=params_lstm,
        )

        lstm_artifact_dir = save_rnn_artifacts(
            model=model_lstm,
            tokenizer=tokenizer,
            model_name="lstm",
            rnn_type="lstm",
            num_classes=num_classes,
            label_mapping=label_mapping,
        )
        log_pyfunc_model(
            model_dir=str(lstm_artifact_dir),
            python_model=RNNPyFunc(),
            artifact_path="model",
            pip_requirements=["mlflow", "pandas", "torch"],
        )

        version_lstm = register_model(
            run_id=run_info.info.run_id,
            artifact_path="model",
            model_name=REGISTERED_MODEL_NAME_LSTM,
        )
        set_model_description(
            REGISTERED_MODEL_NAME_LSTM,
            version_lstm,
            "LSTM recurrent classifier for Banking77 intents.",
        )

    with mlflow.start_run(run_name="GRU") as run_info:
        log_git_commit()
        metrics_gru = train_eval_rnn_model(
            model_gru,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            optimizer_gru,
            criterion,
            DEVICE,
        )
        params_gru = {
            "model_type": "GRU",
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "device": str(DEVICE),
        }
        log_experiment(
            model=None,
            metrics={"test_f1_macro": metrics_gru["f1_macro"]},
            params=params_gru,
        )

        gru_artifact_dir = save_rnn_artifacts(
            model=model_gru,
            tokenizer=tokenizer,
            model_name="gru",
            rnn_type="gru",
            num_classes=num_classes,
            label_mapping=label_mapping,
        )
        log_pyfunc_model(
            model_dir=str(gru_artifact_dir),
            python_model=RNNPyFunc(),
            artifact_path="model",
            pip_requirements=["mlflow", "pandas", "torch"],
        )

        version_gru = register_model(
            run_id=run_info.info.run_id,
            artifact_path="model",
            model_name=REGISTERED_MODEL_NAME_GRU,
        )
        set_model_description(
            REGISTERED_MODEL_NAME_GRU,
            version_gru,
            "GRU recurrent classifier for Banking77 intents.",
        )

if __name__ == "__main__":
    run()
