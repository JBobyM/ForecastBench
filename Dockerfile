# Serving image: FastAPI + precomputed forecasts. No ML libraries, so it stays small.
FROM python:3.12-slim
WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY app/ app/
COPY artifacts/serve/ artifacts/serve/
ENV PORT=8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
