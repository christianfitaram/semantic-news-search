#!/usr/bin/env sh
set -eu

MODEL_DIR="${HF_LOCAL_MODELS_DIR:-/app/models}"
BOOTSTRAP_MODELS="${BOOTSTRAP_MODELS:-1}"

if [ "$BOOTSTRAP_MODELS" = "1" ]; then
  if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "[bootstrap] No local models found in $MODEL_DIR. Downloading..."
    mkdir -p "$MODEL_DIR"
    python /app/scripts/bootstrap_models.py
    echo "[bootstrap] Models downloaded."
  else
    echo "[bootstrap] Found existing models in $MODEL_DIR. Skipping download."
  fi
else
  echo "[bootstrap] BOOTSTRAP_MODELS=0. Skipping model download."
fi

exec "$@" --workers "${UVICORN_WORKERS:-2}"
