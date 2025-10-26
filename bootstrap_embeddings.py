# bootstrap_embeddings.py
import os, math
from tqdm import tqdm
from textwrap import wrap
from dotenv import load_dotenv
from pymongo import MongoClient
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------
# Load environment
# -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
POSTGRES_URI = os.getenv("POSTGRES_URI")

# -----------------
# Connections
# -----------------
mongo = MongoClient(MONGO_URI)
news_coll = mongo.news.articles  # adjust collection name if needed
pg = psycopg2.connect(POSTGRES_URI)
pg.autocommit = True

# -----------------
# Model
# -----------------
model = SentenceTransformer("BAAI/bge-m3")

# -----------------
# Parameters
# -----------------
CHUNK_SIZE = 700  # ~500–800 tokens
OVERLAP = 100
BATCH_SIZE = 16  # embeddings per batch


# -----------------
# Helper functions
# -----------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks


def insert_embedding(cur, article, chunk, emb):
    cur.execute("""
        INSERT INTO news_embeddings
        (article_id, title, content, topic, sentiment_label,
         sentiment_score, source, scraped_at, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        article["data"]["id"],
        article["data"]["title"],
        chunk,
        article["data"].get("topic"),
        article["data"]["sentiment"]["label"],
        article["data"]["sentiment"]["score"],
        article["data"]["source"],
        article["data"]["scrapedAt"],
        list(map(float, emb))
    ))


# -----------------
# Main loop
# -----------------
with pg.cursor() as cur:
    articles = news_coll.find({"data.cleaned": True})
    for article in tqdm(articles, desc="Embedding articles"):
        text = article["data"].get("text")
        if not text:
            continue
        chunks = chunk_text(text)
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embs = model.encode(batch, normalize_embeddings=True)
            for chunk, emb in zip(batch, embs):
                insert_embedding(cur, article, chunk, emb)
