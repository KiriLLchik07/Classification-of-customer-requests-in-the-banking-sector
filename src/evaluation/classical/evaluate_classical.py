from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_classical(model, X, y) -> dict[str, float]:
    """Инференс модели классического машинного обучения на выборке валидации или тестирования."""
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(y, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
    }
