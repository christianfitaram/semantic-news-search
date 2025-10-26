CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE news_embeddings (
    id SERIAL PRIMARY KEY,
    article_id TEXT,
    title TEXT,
    content TEXT,
    topic TEXT,
    sentiment_label TEXT,
    sentiment_score FLOAT,
    source TEXT,
    scraped_at TIMESTAMP,
    embedding vector(1024)
);

CREATE INDEX news_embeddings_ivfflat
ON news_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
