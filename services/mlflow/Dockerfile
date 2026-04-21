FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==3.8.1 psycopg2-binary boto3
COPY services/mlflow/start_mlflow.sh /app/start_mlflow.sh
RUN chmod +x /app/start_mlflow.sh

EXPOSE 5000
