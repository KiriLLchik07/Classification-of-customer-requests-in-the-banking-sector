#!/bin/sh
set -eu

: "${MINIO_ENDPOINT:=http://minio:9000}"
: "${MINIO_ROOT_USER:?MINIO_ROOT_USER is required}"
: "${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD is required}"
: "${MLFLOW_S3_BUCKET:=mlflow-artifacts}"

until /usr/bin/mc alias set local "${MINIO_ENDPOINT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"; do
  echo "Waiting for MinIO..."
  sleep 2
done

/usr/bin/mc mb --ignore-existing "local/${MLFLOW_S3_BUCKET}"
/usr/bin/mc version enable "local/${MLFLOW_S3_BUCKET}"

echo "MinIO bucket is ready: ${MLFLOW_S3_BUCKET}"
