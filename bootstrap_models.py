# bootstrap_models.py
"""
Download and cache the embedding and reranker models locally so the API can run
without live access to huggingface.co.

Usage:
    python bootstrap_models.py

Environment variables (all optional):
    EMBEDDING_MODEL   - Hugging Face repo id or local path (default: BAAI/bge-m3)
    RERANKER_MODEL    - Hugging Face repo id or local path (default: BAAI/bge-reranker-base)
    MODEL_CACHE_DIR   - Base directory to store downloaded models (default: ./models)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


def resolve_target(model_id: str, cache_root: Path) -> Path:
    """Choose a local directory for the model, respecting existing paths."""
    path_candidate = Path(model_id)
    if path_candidate.exists():
        return path_candidate
    slug = model_id.rstrip("/").split("/")[-1] or "model"
    return cache_root / slug


def ensure_model(model_id: str, target_dir: Path) -> None:
    """Download model if not already present."""
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[skip] Found existing files for '{model_id}' at {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching '{model_id}' into {target_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[ok] Downloaded '{model_id}' to {target_dir}")


def main() -> int:
    embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    rerank_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    cache_root = Path(os.getenv("MODEL_CACHE_DIR", "models")).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Embedding model: {embed_model}")
    print(f"Reranker model:  {rerank_model}")
    print(f"Cache root:      {cache_root}")

    try:
        ensure_model(embed_model, resolve_target(embed_model, cache_root))
        ensure_model(rerank_model, resolve_target(rerank_model, cache_root))
    except Exception as exc:
        print(f"[error] Failed to download models: {exc}", file=sys.stderr)
        return 1

    print("Done. Update your .env with absolute paths if running offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
