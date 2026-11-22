# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/storage/hf-cache

WORKDIR /app/backend

# Set INSTALL_LOCAL_MODELS=false for a slim image that uses EMBEDDING_PROVIDER=openai or hash.
ARG INSTALL_LOCAL_MODELS=true

COPY backend/requirements.txt backend/requirements-local-models.txt ./
RUN pip install -r requirements.txt \
 && if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install -r requirements-local-models.txt; \
    fi

# Pre-download the default embedding model into the image (copied into the data volume on first run)
# so the first upload does not wait on a download. Skipped silently if offline at build time.
ARG PRELOAD_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN if [ "$INSTALL_LOCAL_MODELS" = "true" ] && [ -n "$PRELOAD_EMBEDDING_MODEL" ]; then \
      python -c "from sentence_transformers import SentenceTransformer as S; S('$PRELOAD_EMBEDDING_MODEL')" \
      || echo "Model preload skipped; it will download on first use"; \
    fi

COPY backend/app ./app
COPY backend/tests ./tests
COPY backend/pytest.ini ./pytest.ini
COPY evaluation/dataset /app/evaluation/dataset

RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/storage /app/evaluation/reports \
 && chown -R appuser:appuser /app
USER appuser

ENV DATA_DIR=/app/storage \
    EVALUATION_DATASET_PATH=/app/evaluation/dataset/benchmark.json \
    EVALUATION_REPORTS_DIR=/app/storage/evaluation-reports

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
