import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

class PyFunc(mlflow.pyfunc.PythonModel):
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

        self.label_mapping = context.artifacts["label_mapping"]

    def predict(self, model_input: pd.DataFrame):
        texts = model_input["texts"].tolist()

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        return preds
