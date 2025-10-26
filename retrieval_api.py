# retrieval_api.py
import os, time
from typing import List, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
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


class SearchResult(BaseModel):
    title: Optional[str]
    source: Optional[str]
    topic: Optional[str]
    scraped_at: Optional[str]
    content_preview: Optional[str]
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    score: float


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
        SELECT title, source, topic, scraped_at, content, sentiment_label, sentiment_score,
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
            topic=r[2],
            scraped_at=r[3].isoformat() if r[3] else None,
            content_preview=(r[4][:220] + "...") if r[4] else None,
            sentiment_label=r[5],
            sentiment_score=float(r[6]) if r[6] else None,
            score=float(r[7]),
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
