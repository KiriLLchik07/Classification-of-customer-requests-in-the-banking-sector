import pandas as pd

class InferenceService:
    def __init__(self, model):
        self.model = model

    def predict(self, text: str):
        df = pd.DataFrame({"text": [text]})
        pred = self.model.predict(df)[0]

        return pred
