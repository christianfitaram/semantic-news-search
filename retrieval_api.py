# retrieval_api.py
import os, time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
import psycopg2
from dotenv import load_dotenv
import numpy as np

# -----------------
# Load environment
# -----------------
load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# -----------------
# Init models
# -----------------
print(f"Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"Loading reranker model: {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL)

# -----------------
# Connect Postgres
# -----------------
pg = psycopg2.connect(POSTGRES_URI)
pg.autocommit = True

# -----------------
# FastAPI setup
# -----------------
app = FastAPI(title="Semantic News Search API", version="1.0")


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


@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., description="User query text"),
    k: int = Query(10, description="Number of results to return"),
    rerank: bool = Query(True, description="Enable cross-encoder reranking"),
):
    t0 = time.time()

    # 1. Embed query
    query_emb = embedder.encode([q], normalize_embeddings=True)[0].tolist()

    # 2. Retrieve top-k candidates by cosine similarity
    cur = pg.cursor()
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

    if not rows:
        return []

    results = [
        dict(
            title=r[0],
            source=r[1],
            url=r[2],
            topic=r[3],
            scraped_at=r[4].isoformat() if r[4] else None,
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


def insert_embedding(cur, article: ArticlePayload, chunk: str, embedding: np.ndarray) -> None:
    sentiment = article.resolve_sentiment()
    cur.execute(
        """
        INSERT INTO news_embeddings
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
def ingest_newshook(article: ArticlePayload):
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
    with pg.cursor() as cur:
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            embeddings = embedder.encode(batch, normalize_embeddings=True)
            for chunk, emb in zip(batch, embeddings):
                try:
                    insert_embedding(cur, article, chunk, emb)
                    inserted += 1
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to persist embedding chunk: {exc}",
                    )

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
