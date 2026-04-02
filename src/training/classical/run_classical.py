from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation.classical.evaluate_classical import evaluate_classical

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=50000,
        sublinear_tf=True,
    )

def build_models() -> dict[str, object]:
    tfidf = build_tfidf()
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", tfidf),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", build_tfidf()),
                ("classifier", MultinomialNB()),
            ]
        ),
        "catboost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.1,
            depth=8,
            loss_function="MultiClass",
            verbose=False,
        ),
    }


def run() -> pd.DataFrame:
    train_df = pd.read_csv(DATA_DIR / "train_df.csv")
    val_df = pd.read_csv(DATA_DIR / "val_df.csv")

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()

    models = build_models()
    results: dict[str, dict[str, float]] = {}

    models["logistic_regression"].fit(X_train, y_train)
    results["logistic_regression"] = evaluate_classical(models["logistic_regression"], X_val, y_val)

    models["multinomial_nb"].fit(X_train, y_train)
    results["multinomial_nb"] = evaluate_classical(models["multinomial_nb"], X_val, y_val)

    tfidf = build_tfidf()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    models["catboost"].fit(X_train_tfidf, y_train)
    results["catboost"] = evaluate_classical(models["catboost"], X_val_tfidf, y_val)

    return pd.DataFrame(results).T.sort_values("f1_macro", ascending=False)


if __name__ == "__main__":
    print(run())
