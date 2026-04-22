# Классификация клиентских запросов в банковской сфере

## Описание проекта

Данный проект представляет собой **end-to-end ML-систему** для классификации текстовых запросов клиентов банка по определенному типу.

Проект построен на основе открытого датасета [**PolyAI/banking77**](https://huggingface.co/datasets/PolyAI/banking77) и охватывает такие этапы разработки как:

- исследование данных;
- обучение и сравнение моделей;
- трекинг экспериментов;
- деплой модели;
- backend API;
- frontend demo;
- контейнеризация и базовые MLOps-практики.

**Задача проекта** - автоматически определять тип клиентского запроса, чтобы маршрутизировать его в нужный сервис поддержки, автоматизировать ответы, сократить нагрузку на операторов.

## Датасет

Используется датасет **Banking77**, который имеет 77 классов (типов клиентских обращений) и состоит из 13 083 текстовых запросов. Датасет представляет собой реальную банковскую тематику.

## ML-подходы, реализованные в проекте

В проекте сравниваются **несколько классов моделей**:

### Классический Machine Learning

Текст очищается, векторизуется с помощью TF-IDF, затем на основе векторного представления текста используются следующие алгоритмы машинного обучения:

- `Logistic Regression` **(baseline модель)**;
- `Multinomial Naive Bayes`;
- `CatBoost Classifier`.

### Deep Learning

В качестве продвинутого варианта решения поставленной задачи используются архитектуры рекуррентных нейронных сетей: `LSTM` и `GRU`.

### Transformers

Являются наиболее эффективной технологией в решении задач классификации текстовых документов. Были взяты модели `BERT` и `DistilBERT`, которые были дообучены под решение поставленной задачи.

## Архитектура решения

Все эксперименты логируются и сравниваются с помощью **MLflow**. В качестве backend store для хранения метаданных моделей и экспериментов используется **PostgreSQL**. В качестве artifacts store для хранения моделей используется S3-совместимое хранилище **MinIO**.

### Общая архитектура

```
Frontend (Streamlit)
        ↓
FastAPI Backend
        ↓
ML Service Layer
        ↓
MLflow Model Registry
    ↓             ↓
MinIO S3      PostgreSQL
```

![Общая архитектура](assets/arhitecture_schema.png)

### Pipeline обучения

```
Датасет → Предобработка → Feature Engineering → Обучение моделей → Предсказания → MLflow Tracking → Регистрация моделей
```

### Pipeline инференса

```
Запрос клиента → API → Модель → Предсказания → Ответ
```

## Технологический стек

### Основные технологии

- **Python 3.11**
- **Poetry**
- **scikit-learn**
- **PyTorch**
- **Transformers**
- **MLflow**
- **FastAPI**
- **PostgreSQL**
- **MinIO S3**
- **Docker / docker-compose**
- **Streamlit**

## Структура репозитория

```
Classification-of-customer-requests-in-the-banking-sector/
│
├── .github/workflows/  # CI (запуск тестов)
├── config/             # Конфигурация (settings, env, label mapping)
├── data/               # train, val и test датасеты
├── notebooks/          # Исследования и эксперименты
├── services/           # Сервисный слой приложения
│   ├── backend/        # FastAPI backend
│   ├── frontend/       # Streamlit UI
│   ├── mlflow/         # MLflow server (Docker + start script)
│   └── minio/          # Инициализация S3 bucket для артефактов
├── src/                # ML-ядро проекта
│   ├── data/           # Датасеты и dataloader-обертки
│   ├── features/       # Токенизация и collate-функции
│   ├── training/       # Скрипты обучения (classical/rnn/transformers)
│   ├── evaluation/     # Скрипты оценки и error analysis
│   ├── models/         # Архитектуры моделей
│   └── mlops/          # MLflow tracking/registry + pyfunc packaging
├── tests/              # API и unit-тесты
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Краткая инструкция по запуску проекта

1. Клонирование репозитория:

```bash
git clone https://github.com/KiriLLchik07/Classification-of-customer-requests-in-the-banking-sector.git
cd Classification-of-customer-requests-in-the-banking-sector

# установка зависимостей
poetry install
```

2. Создание файла config/environment.env. Пример:

```env
PROJECT_NAME=Banking77 ML Service
DEBUG=false

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
MODEL_NAME=Banking77_DistilBERT
MODEL_ALIAS=production

POSTGRES_USER=mlflow_db
POSTGRES_PASSWORD=mlflow_db
POSTGRES_DB=mlflow_db
POSTGRES_HOST=mlflow_postgres
POSTGRES_PORT=5432

AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
AWS_DEFAULT_REGION=us-east-1
AWS_EC2_METADATA_DISABLED=true

MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_ENDPOINT=http://minio:9000
MLFLOW_S3_BUCKET=mlflow-artifacts

MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
MLFLOW_ALLOWED_HOSTS=*
MLFLOW_CORS_ALLOWED_ORIGINS=*
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlflow_db:mlflow_db@mlflow_postgres:5432/mlflow_db
MLFLOW_ARTIFACTS_DESTINATION=s3://mlflow-artifacts
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
MLFLOW_RUN_DB_UPGRADE=false

BACKEND_URL=http://backend:8000

DEVICE=cpu
MAX_TEXT_LENGTH=512
MIN_TEXT_LENGTH=3
```

3. Запустить сервисы:

```bash
docker compose up --build
```

**ВАЖНО!**

После запуска сервисов у вас будут пустые `backend store` и `artifacts store`. Для того, чтобы можно было успешно пользоваться системой, вам необходимо запустить скрипты обучения моделей из корня проекта:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
poetry run python -m src.training.classical.run_classical
poetry run python -m src.training.rnn.train_rnn
poetry run python -m src.training.transformers.run_transformer_bert
poetry run python -m src.training.transformers.run_transformer_distilbert
```

Если у вас Windows:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python -m src.training.classical.run_classical
poetry run python -m src.training.rnn.train_rnn
poetry run python -m src.training.transformers.run_transformer_bert
poetry run python -m src.training.transformers.run_transformer_distilbert

```

После этого модели будут зарегистрированы в **MLflow Registry**, и вы сможете пользоваться системой.

**Подключение к сервисам:**

- localhost:5433 - PostgreSQL (через PgAdmin или любой другой клиент).
- http://localhost:9000 - MinIO S3.
- http://localhost:5000 - MLflow UI.
- http://localhost:8000 - FastAPI backend.
- http://localhost:8501 - Streamlit Web.

## Результаты

В проекте проводится сравнение всех подходов по метрике F1-macro, анализ ошибок моделей, а также выбор наилучшей модели.

Модели `Logistic Regression (baseline)`, `LSTM`, `GRU`, `BERT`, `DistilBERT` регистрируются в **MLflow Model Registry** и используются в API.

### Таблица сравнения моделей по метрике F1-macro.

| Модель                             | F1-macro    | MLflow Registry alias |
| ---------------------------------- | ----------- | --------------------- |
| **Logistic Regression (baseline)** | **0.84454** | **@baseline**         |
| Multinomial Naive Bayes            | 0.76404     | - (not in registry)   |
| CatBoost Classifier                | 0.77500     | - (not in registry)   |
| LSTM                               | 0.86493     | @reserve              |
| GRU                                | 0.87224     | @reserve              |
| **BERT**                           | **0.92526** | **@production**       |
| **DistilBERT**                     | **0.92010** | **@production**       |

## Streamlit Web

<div style="text-align:center; padding: 10px; font-weight:bold">
Главный экран приложения
</div>

![Домашний экран](assets/home_page.png)

<div style="text-align:center; padding: 10px; font-weight:bold">
Экран с предсказанием модели
</div>

![Экран с предсказанием](assets/prediction_section2.png)

<div style="text-align:center; padding: 10px; font-weight:bold">
Экран с информацией о моделях
</div>

![Экран с информацией о моделях](assets/models_info.png)
