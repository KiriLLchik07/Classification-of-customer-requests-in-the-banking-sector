from pathlib import Path
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation.classical.evaluate_classical import evaluate_classical
from src.mlops.mlflow.dataset import file_md5
from src.mlops.mlflow.logging import log_git_commit, seed_everything
from src.mlops.mlflow.registry import register_model, set_model_description
from src.mlops.mlflow.tracking import setup_mlflow
from src.mlops.packaging.log_pyfunc_model import ClassicPyFunc, log_pyfunc_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "local_models" / "classical"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REGISTERED_BASELINE_NAME = "Banking77_LogisticRegression"
BASELINE_DESCRIPTION = (
    "Logistic Regression baseline trained with TF-IDF features "
    "for Banking77 intent classification."
)

def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=50000,
        sublinear_tf=True,
    )

def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", build_tfidf()),
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

def log_common_params(train_path: Path, val_path: Path, model_family: str, algorithm: str) -> None:
    mlflow.log_params(
        {
            "model_family": model_family,
            "algorithm": algorithm,
            "train_df_md5": file_md5(train_path),
            "val_df_md5": file_md5(val_path),
            "python_model_flavor": "sklearn",
        }
    )
    log_git_commit()

def build_label_mapping(train_df: pd.DataFrame) -> dict[int, str]:
    labels = sorted(train_df["label"].unique().tolist())
    return {int(label): str(label) for label in labels}

def run() -> pd.DataFrame:
    seed_everything(42)
    setup_mlflow()

    train_path = DATA_DIR / "train_df.csv"
    val_path = DATA_DIR / "val_df.csv"
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    x_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()
    label_mapping = build_label_mapping(train_df)

    models = build_models()
    results: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name="Logistic_Regression_Baseline") as run_info:
        log_common_params(train_path, val_path, model_family="classical", algorithm="logistic_regression")
        mlflow.log_params(
            {
                "tfidf_ngram_range": "(1, 2)",
                "tfidf_min_df": 2,
                "tfidf_max_df": 0.9,
                "tfidf_max_features": 50000,
                "tfidf_sublinear_tf": True,
                "classifier_max_iter": 1000,
                "classifier_class_weight": "balanced",
            }
        )

        models["logistic_regression"].fit(x_train, y_train)
        logistic_artifact_dir = ARTIFACT_DIR / "logistic_regression"
        logistic_artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models["logistic_regression"], logistic_artifact_dir / "pipeline.joblib")
        with open(logistic_artifact_dir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)

        results["logistic_regression"] = evaluate_classical(models["logistic_regression"], x_val, y_val)
        mlflow.log_metrics(results["logistic_regression"])
        log_pyfunc_model(
            model_dir=str(logistic_artifact_dir),
            python_model=ClassicPyFunc(),
            artifact_path="model",
            pip_requirements=["mlflow", "pandas", "scikit-learn", "joblib"],
        )

        version = register_model(
            run_id=run_info.info.run_id,
            artifact_path="model",
            model_name=REGISTERED_BASELINE_NAME,
        )
        set_model_description(
            model_name=REGISTERED_BASELINE_NAME,
            version=version,
            description=BASELINE_DESCRIPTION,
        )

    with mlflow.start_run(run_name="Multinomial_Naive_Bayes"):
        log_common_params(train_path, val_path, model_family="classical", algorithm="multinomial_nb")
        mlflow.log_params(
            {
                "tfidf_ngram_range": "(1, 2)",
                "tfidf_min_df": 2,
                "tfidf_max_df": 0.9,
                "tfidf_max_features": 50000,
                "tfidf_sublinear_tf": True,
            }
        )

        models["multinomial_nb"].fit(x_train, y_train)
        results["multinomial_nb"] = evaluate_classical(models["multinomial_nb"], x_val, y_val)
        mlflow.log_metrics(results["multinomial_nb"])
        mlflow.sklearn.log_model(models["multinomial_nb"], artifact_path="model")

    with mlflow.start_run(run_name="CatBoost_Classifier"):
        log_common_params(train_path, val_path, model_family="classical", algorithm="catboost")
        mlflow.log_params(
            {
                "iterations": 300,
                "learning_rate": 0.1,
                "depth": 8,
                "loss_function": "MultiClass",
            }
        )

        tfidf = build_tfidf()
        x_train_tfidf = tfidf.fit_transform(x_train)
        x_val_tfidf = tfidf.transform(x_val)
        models["catboost"].fit(x_train_tfidf, y_train)
        results["catboost"] = evaluate_classical(models["catboost"], x_val_tfidf, y_val)
        mlflow.log_metrics(results["catboost"])
        mlflow.sklearn.log_model(models["catboost"], artifact_path="model")

    return pd.DataFrame(results).T.sort_values("f1_macro", ascending=False)

if __name__ == "__main__":
    print(run())
