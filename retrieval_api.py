# retrieval_api.py
import hmac
import hashlib
import inspect
import json
import os, time
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import Header, HTTPException, status, Request

from fastapi import FastAPI, HTTPException, Query
from fastapi.params import Depends
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
import psycopg2
from dotenv import load_dotenv
import numpy as np

# -----------------
# Load environment
# -----------------
load_dotenv()

API_SECRET = os.getenv("API_KEY", "")


# Dependency function to verify API key

def _env_flag(name: str, default: Optional[bool] = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.lower() in ("1", "true", "yes", "y", "on")


POSTGRES_URI = os.getenv("POSTGRES_URI")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
HF_LOCAL_MODELS_DIR = os.getenv("HF_LOCAL_MODELS_DIR")
HF_MODEL_CACHE_DIR = os.getenv("HF_MODEL_CACHE_DIR")

# Derive offline flag: explicit env wins; otherwise inherit TRANSFORMERS_OFFLINE
HF_LOCAL_FILES_ONLY = _env_flag(
    "HF_LOCAL_FILES_ONLY",
    default=_env_flag("TRANSFORMERS_OFFLINE", default=False),
)

if HF_LOCAL_FILES_ONLY:
    # Prevent accidental network calls on air-gapped hosts
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# -----------------
# Init models
# -----------------
def _resolve_model_path(model_name: str) -> str:
    """
    If HF_LOCAL_MODELS_DIR is set, try to load the model from that directory
    (useful on air‑gapped hosts where models are pre-downloaded). We check both
    a subdir matching the full repo id and a subdir matching the final slug.
    Otherwise return the original identifier so HF hub can fetch it when allowed.
    """
    # Absolute or already-local path: use as-is
    if os.path.isdir(model_name):
        return model_name

    if HF_LOCAL_MODELS_DIR:
        full = os.path.join(HF_LOCAL_MODELS_DIR, model_name)
        slug = os.path.join(
            HF_LOCAL_MODELS_DIR, model_name.rstrip("/").split("/")[-1]
        )
        for candidate in (full, slug):
            if os.path.isdir(candidate):
                return candidate
    return model_name


def _load_sentence_model(model_name: str):
    resolved = _resolve_model_path(model_name)
    kwargs = {"cache_folder": HF_MODEL_CACHE_DIR}
    if "local_files_only" in inspect.signature(SentenceTransformer.__init__).parameters:
        kwargs["local_files_only"] = HF_LOCAL_FILES_ONLY
    try:
        return SentenceTransformer(resolved, **kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"Embedding model '{resolved}' not available locally "
            f"(local_files_only={HF_LOCAL_FILES_ONLY}). "
            "Pre-download the model into HF cache or set HF_LOCAL_MODELS_DIR "
            f"to a directory containing '{model_name}' (or provide an absolute path "
            "via EMBEDDING_MODEL)."
        ) from exc


def _load_reranker_model(model_name: str):
    resolved = _resolve_model_path(model_name)
    kwargs = {"cache_folder": HF_MODEL_CACHE_DIR}
    if "local_files_only" in inspect.signature(CrossEncoder.__init__).parameters:
        kwargs["local_files_only"] = HF_LOCAL_FILES_ONLY
    try:
        return CrossEncoder(resolved, **kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"Reranker model '{resolved}' not available locally "
            f"(local_files_only={HF_LOCAL_FILES_ONLY}). "
            "Pre-download the model into HF cache or set HF_LOCAL_MODELS_DIR "
            f"to a directory containing '{model_name}' (or provide an absolute path "
            "via RERANKER_MODEL)."
        ) from exc


print(
    f"Loading embedding model: {EMBED_MODEL} "
    f"(local_files_only={HF_LOCAL_FILES_ONLY}, cache_dir={HF_MODEL_CACHE_DIR})"
)
embedder = _load_sentence_model(EMBED_MODEL)

print(
    f"Loading reranker model: {RERANK_MODEL} "
    f"(local_files_only={HF_LOCAL_FILES_ONLY}, cache_dir={HF_MODEL_CACHE_DIR})"
)
reranker = _load_reranker_model(RERANK_MODEL)

# -----------------
# Connect Postgres
# -----------------
pg = psycopg2.connect(POSTGRES_URI)
pg.autocommit = True
pg_pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=POSTGRES_URI)


def get_db_conn():
    conn = pg_pool.getconn()
    conn.autocommit = True
    return conn


# -----------------
# FastAPI setup
# -----------------
app = FastAPI(title="Semantic News Search API", version="1.0")

async def verify_signature(
    request: Request,
    x_signature: str = Header(..., description="HMAC signature of request body")
):
    """
    Verifies the X-Signature header using HMAC-SHA256.
    Expected format: sha256=<hex_digest>
    """
    body = await request.body()
    print(f"Verifying signature for body: {body.decode().encode()}")
    print(f"Received signature: {x_signature}")
    # Compute expected signature
    computed_signature = hmac.new(
        API_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    # Verify signature (constant-time comparison to prevent timing attacks)
    expected = f"sha256={computed_signature}"
    
    if not hmac.compare_digest(x_signature, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature"
        )
    
    return True

class SentimentPayload(BaseModel):
    label: Optional[str] = None
    score: Optional[float] = None


class ArticlePayload(BaseModel):
    """Payload expected from the news scraper webhook."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    article_id: Optional[str] = None
    mongo_id: Optional[str] = Field(default=None, alias="_id")
    title: Optional[str] = None
    text: str
    topic: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    scraped_at: Optional[datetime] = None
    sentiment: Optional[SentimentPayload] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    table: str = "news_embeddings"

    def resolve_article_id(self) -> str:
        candidates = [
            self.article_id,
            self.mongo_id,
            (self.data or {}).get("id") if isinstance(self.data, dict) else None,
            (self.data or {}).get("_id") if isinstance(self.data, dict) else None,
        ]
        for cand in candidates:
            if cand:
                return str(cand)
        return ""

    def resolve_sentiment(self) -> SentimentPayload:
        if self.sentiment:
            label = self.sentiment.label or self.sentiment_label
            score = (
                float(self.sentiment.score)
                if self.sentiment.score is not None
                else self.sentiment_score
            )
        else:
            label = self.sentiment_label
            score = self.sentiment_score
        return SentimentPayload(label=label, score=score)

    def resolve_topic(self) -> Optional[str]:
        if self.topic:
            return self.topic
        if isinstance(self.data, dict):
            return self.data.get("topic")
        return None

    def resolve_source(self) -> Optional[str]:
        if self.source:
            return self.source
        if isinstance(self.data, dict):
            return self.data.get("source")
        return None

    def resolve_url(self) -> Optional[str]:
        if self.url:
            return self.url
        if isinstance(self.data, dict):
            return self.data.get("url")
        return None


class WebhookResponse(BaseModel):
    article_id: str
    chunks_inserted: int


class SearchResult(BaseModel):
    title: Optional[str]
    source: Optional[str]
    url: Optional[str]
    topic: Optional[str]
    scraped_at: Optional[str]
    content_preview: Optional[str]
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]


@app.get("/health")
def healthcheck(x_api_key: str = Header(..., description="API key for access")):
    """Lightweight health probe used by process managers."""
    
    """GET endpoints can use simple API key."""
    if x_api_key not in {os.getenv("API_KEY")}:
        raise HTTPException(status_code=401, detail="Invalid API key")
    t0 = time.time()
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database check failed: {exc}")
    finally:
        if "conn" in locals():
            pg_pool.putconn(conn)

    return {
        "status": "ok",
        "database": "ok",
        "embedding_model": EMBED_MODEL,
        "reranker_model": RERANK_MODEL,
    }


@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., description="User query text"),
    k: int = Query(10, description="Number of results to return"),
    rerank: bool = Query(True, description="Enable cross-encoder reranking"),
    x_api_key: str = Header(..., description="API key for access"),
    date_order: str = Query(
        "desc",
        description="Sort results by scraped_at; use 'asc' or 'desc' (default).",
    ),
):
    """GET endpoints can use simple API key."""
    if x_api_key not in {os.getenv("API_KEY")}:
        raise HTTPException(status_code=401, detail="Invalid API key")
    t0 = time.time()
    date_order = (date_order or "desc").lower()
    if date_order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="Parameter 'date_order' must be 'asc' or 'desc'.")

    # 1. Embed query
    query_emb = embedder.encode([q], normalize_embeddings=True)[0].tolist()

    # 2. Retrieve top-k candidates by cosine similarity
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT title, source, url, topic, scraped_at, content, sentiment_label, sentiment_score,
               1 - (embedding <=> %s::vector) AS score
        FROM news_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_emb, query_emb, k * (5 if rerank else 1)),
    )
    rows = cur.fetchall()
    cur.close()
    pg_pool.putconn(conn)

    if not rows:
        return []

    results = [
        dict(
            title=r[0],
            source=r[1],
            url=r[2],
            topic=r[3],
            scraped_at=r[4].isoformat() if r[4] else None,
            _scraped_at_dt=r[4],
            content_preview=(r[5][:220] + "...") if r[5] else None,
            sentiment_label=r[6],
            sentiment_score=float(r[7]) if r[7] is not None else None,
        )
        for r in rows
    ]

    # 3. Optional rerank (cross-encoder)
    if rerank:
        pairs = [(q, r["content_preview"] or "") for r in results]
        scores = reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["score"] = float(s)
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:k]

    results.sort(
        key=lambda r: r.get("_scraped_at_dt") or datetime.min,
        reverse=date_order != "asc",
    )
    for r in results:
        r.pop("_scraped_at_dt", None)

    print(f"Query '{q}' processed in {time.time()-t0:.2f}s")
    return results


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start_idx in range(0, len(words), step):
        chunk = " ".join(words[start_idx : start_idx + chunk_size])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks


def insert_embedding(cur, article: ArticlePayload, chunk: str, embedding: np.ndarray, table: str) -> None:
    sentiment = article.resolve_sentiment()
    cur.execute(
        f"""
        INSERT INTO {table}
        (article_id, title, url, content, topic, sentiment_label,
         sentiment_score, source, scraped_at, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            article.resolve_article_id(),
            article.title,
            article.resolve_url(),
            chunk,
            article.resolve_topic(),
            sentiment.label,
            sentiment.score,
            article.resolve_source(),
            article.scraped_at,
            np.asarray(embedding, dtype=float).tolist(),
        ),
    )

@app.post("/webhook/news", response_model=WebhookResponse, status_code=201)
def ingest_newshook(
    article: ArticlePayload,
    x_signature: str = Header(...),
    verified: bool = Depends(verify_signature)  # Add this
):
    """
    Accept a news document from the scraper, chunk + embed its text,
    and persist the resulting vectors into Postgres.
    """

    text = article.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must contain content.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Content is too short to chunk; include more text.",
        )

    inserted = 0
    conn = get_db_conn()
    with conn.cursor() as cur:
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            embeddings = embedder.encode(batch, normalize_embeddings=True)
            for chunk, emb in zip(batch, embeddings):
                try:
                    insert_embedding(cur, article, chunk, emb, article.table)
                    inserted += 1
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to persist embedding chunk: {exc}",
                    )
    pg_pool.putconn(conn)

    article_id = article.resolve_article_id()
    if not article_id:
        article_id = "unknown"
    return WebhookResponse(article_id=article_id, chunks_inserted=inserted)

# --- New endpoint: serve embeddings to Next.js backend ---
class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]


@app.post("/api/embeddings", response_model=EmbeddingResponse)
def get_embedding(req: EmbeddingRequest):
    """Return normalized embedding vector for provided text.

    This endpoint is intended for the Next.js backend to request embeddings
    without directly loading the sentence-transformers model there.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must contain content.")

    try:
        emb = embedder.encode([text], normalize_embeddings=True)[0].tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {exc}")

    return EmbeddingResponse(embedding=emb)
