from pathlib import Path
import pandas as pd
import torch
import json
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Загружает модель один раз при старте приложения
        """

        model_path = context.artifacts["model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        label_path = Path(model_path) / "label_mapping.json"

        with open(label_path) as f:
            self.label_mapping = json.load(f)

        self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}

    def predict(self, context, model_input: pd.DataFrame):
        """
        model_input обязан содержать колонку "text"
        """
        texts = model_input["text"].tolist()

        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        labels = [self.label_mapping[p] for p in preds]

        return pd.DataFrame({"prediction": labels})

def log_pyfunc_model(model_dir: str, artifact_path: str = "model"):
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=PyFuncModel(), 
        artifacts={"model": model_dir},
        pip_requirements=["torch", "transformers", "pandas", "mlflow"]    
    )
