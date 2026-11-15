# Serving image for the NSTI GAS predictor (SYNTHETIC data; not for clinical use).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/__init__.py src/config.py src/features.py src/models.py src/api.py src/
COPY model/served_pipeline.joblib model/served_meta.json model/

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
