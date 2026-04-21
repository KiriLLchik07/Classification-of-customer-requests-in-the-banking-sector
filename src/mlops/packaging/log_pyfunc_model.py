import json
from pathlib import Path
import joblib
import mlflow
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.features.tokenizer import TextTokenizer
from src.models.rnn.lstm_gru import RNNClassifier

def resolve_model_artifact_dir(raw_path: str) -> Path:
    direct = Path(raw_path)
    if direct.exists():
        return direct

    normalized = Path(raw_path.replace("\\", "/"))
    if normalized.exists():
        return normalized

    raise FileNotFoundError(
        f"Model artifact path does not exist. raw='{raw_path}', normalized='{normalized}'"
    )

class TransformerPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = resolve_model_artifact_dir(context.artifacts["model"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        label_path = model_path / "label_mapping.json"
        with open(label_path, encoding="utf-8") as f:
            raw_mapping = json.load(f)
        self.label_mapping = {int(k): v for k, v in raw_mapping.items()}

    def predict(self, context, model_input: pd.DataFrame):
        texts = model_input["text"].astype(str).tolist()
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        labels = [self.label_mapping.get(int(p), str(p)) for p in preds]
        return pd.DataFrame({"prediction": labels})

class ClassicPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_dir = resolve_model_artifact_dir(context.artifacts["model"])
        self.pipeline = joblib.load(model_dir / "pipeline.joblib")

        label_mapping_path = model_dir / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, encoding="utf-8") as f:
                raw_mapping = json.load(f)
            self.label_mapping = {int(k): v for k, v in raw_mapping.items()}
        else:
            self.label_mapping = None

    def predict(self, context, model_input: pd.DataFrame):
        texts = model_input["text"].astype(str).tolist()
        preds = self.pipeline.predict(texts)

        if self.label_mapping:
            labels = [self.label_mapping.get(int(p), str(p)) for p in preds]
        else:
            labels = [str(p) for p in preds]

        return pd.DataFrame({"prediction": labels})

class RNNPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_dir = resolve_model_artifact_dir(context.artifacts["model"])

        checkpoint = torch.load(model_dir / "checkpoint.pt", map_location="cpu")
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(model_dir / "label_mapping.json", encoding="utf-8") as f:
            raw_mapping = json.load(f)
        self.label_mapping = {int(k): v for k, v in raw_mapping.items()}

        self.tokenizer = TextTokenizer()
        self.tokenizer.word2idx = checkpoint["vocab"]
        self.tokenizer.idx2word = {idx: word for word, idx in self.tokenizer.word2idx.items()}

        self.max_len = int(cfg["max_len"])
        self.pad_idx = int(self.tokenizer.word2idx.get(self.tokenizer.pad_token, 0))
        self.unk_idx = int(self.tokenizer.word2idx.get(self.tokenizer.unk_token, 1))

        self.model = RNNClassifier(
            vocab_size=len(self.tokenizer.word2idx),
            embedding_dim=int(cfg["embedding_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_classes=int(cfg["num_classes"]),
            rnn_type=str(cfg["rnn_type"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame):
        texts = model_input["text"].astype(str).tolist()
        encoded_batch: list[list[int]] = []
        lengths: list[int] = []

        for text in texts:
            ids = self.tokenizer.encode(text)
            if not ids:
                ids = [self.unk_idx]
            ids = ids[: self.max_len]
            lengths.append(len(ids))
            padded = ids + [self.pad_idx] * (self.max_len - len(ids))
            encoded_batch.append(padded)

        input_ids = torch.tensor(encoded_batch, dtype=torch.long)
        input_lengths = torch.tensor(lengths, dtype=torch.long)

        with torch.no_grad():
            logits = self.model(input_ids, input_lengths)
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        labels = [self.label_mapping.get(int(p), str(p)) for p in preds]
        return pd.DataFrame({"prediction": labels})

def log_pyfunc_model(
    model_dir: str,
    python_model,
    artifact_path: str = "model",
    pip_requirements: list[str] | None = None,
):
    requirements = pip_requirements or ["mlflow", "pandas"]
    model_dir_normalized = model_dir.replace("\\", "/")
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=python_model,
        artifacts={"model": model_dir_normalized},
        code_paths=["src"],
        pip_requirements=requirements,
    )
