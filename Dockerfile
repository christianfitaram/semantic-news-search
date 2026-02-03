FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf \
    HF_MODEL_CACHE_DIR=/app/.hf \
    TRANSFORMERS_CACHE=/app/.hf \
    HF_LOCAL_MODELS_DIR=/app/models \
    HF_LOCAL_FILES_ONLY=1 \
    BOOTSTRAP_MODELS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Allow PyTorch CPU wheels from the official index (override if needed)
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY retrieval_api.py /app/retrieval_api.py
COPY scripts /app/scripts
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080

ENV UVICORN_WORKERS=2
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "retrieval_api:app", "--host", "0.0.0.0", "--port", "8080"]
