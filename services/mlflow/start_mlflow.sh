#!/bin/sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_ARTIFACTS_DESTINATION:?MLFLOW_ARTIFACTS_DESTINATION is required}"
: "${MLFLOW_DEFAULT_ARTIFACT_ROOT:=${MLFLOW_ARTIFACTS_DESTINATION}}"
: "${MLFLOW_HOST:=0.0.0.0}"
: "${MLFLOW_PORT:=5000}"
: "${MLFLOW_ALLOWED_HOSTS:=*}"
: "${MLFLOW_CORS_ALLOWED_ORIGINS:=*}"
: "${MLFLOW_RUN_DB_UPGRADE:=false}"

if [ "${MLFLOW_RUN_DB_UPGRADE}" = "true" ]; then
  mlflow db upgrade "${MLFLOW_BACKEND_STORE_URI}"
fi

exec mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
  --cors-allowed-origins "${MLFLOW_CORS_ALLOWED_ORIGINS}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --serve-artifacts \
  --artifacts-destination "${MLFLOW_ARTIFACTS_DESTINATION}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
