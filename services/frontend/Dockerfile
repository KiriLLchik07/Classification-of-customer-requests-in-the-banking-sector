FROM python:3.11-slim

WORKDIR /app/services/frontend

COPY services/frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/frontend/ ./
COPY config/ /app/config/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
