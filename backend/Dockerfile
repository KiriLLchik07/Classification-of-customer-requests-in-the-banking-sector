FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.2.1
COPY backend/pyproject.toml backend/poetry.lock* ./

RUN pip install torch

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY backend/app/ ./app/
COPY config/ ./config/

RUN mkdir -p /app/logs

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" ]
