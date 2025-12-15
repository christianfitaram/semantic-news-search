# bootstrap_embeddings.py
import os
from tqdm import tqdm
from dotenv import load_dotenv
from pymongo import MongoClient
import psycopg2
from sentence_transformers import SentenceTransformer
import sys
import traceback

# -----------------
# Load environment
# -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
POSTGRES_URI = os.getenv("POSTGRES_URI")

# Prevent accidental non-persistent runs: require POSTGRES_URI unless explicitly allowed
if not POSTGRES_URI and os.getenv("ALLOW_NO_PERSIST") != "1":
    print("Error: POSTGRES_URI is not set. To avoid accidental non-persisted runs the script requires POSTGRES_URI.", file=sys.stderr)
    print("If you intentionally want to run without persisting, set ALLOW_NO_PERSIST=1 in the environment and re-run.", file=sys.stderr)
    sys.exit(1)

print("Diagnostics: MONGO_URI set:", bool(MONGO_URI))
print("Diagnostics: POSTGRES_URI set:", bool(POSTGRES_URI))

if not MONGO_URI:
    print("Error: MONGO_URI is not set. Set it in your environment or .env file.", file=sys.stderr)
    sys.exit(1)

# -----------------
# Connections
# -----------------
try:
    mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # force connection check
    mongo.server_info()
except Exception as e:
    print("Error connecting to MongoDB: ", str(e), file=sys.stderr)
    sys.exit(1)

MONGO_DB = os.getenv("MONGO_DB", "agents")
MONGO_COLL = os.getenv("MONGO_COLL", "articles")

print("Using MongoDB: {}.{}".format(MONGO_DB, MONGO_COLL))

news_coll = mongo[MONGO_DB][MONGO_COLL]  # adjust collection name if needed

# Postgres connection is optional for diagnostics; continue if it fails but warn
pg = None
try:
    if POSTGRES_URI:
        pg = psycopg2.connect(POSTGRES_URI)
        pg.autocommit = True
        print("Postgres connection: OK")
    else:
        print("POSTGRES_URI not set; DB writes will be skipped.")
except Exception as e:
    print("Warning: could not connect to Postgres:", str(e))
    pg = None

# If a POSTGRES_URI was provided but connection failed, exit to avoid silent non-persistence
if POSTGRES_URI and pg is None:
    print("Error: POSTGRES_URI is set but the script could not connect to Postgres. Exiting to avoid lost embeddings.", file=sys.stderr)
    sys.exit(1)

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


def resolve_article_id(article) -> str:
    """Best-effort resolution of an article id from multiple possible fields."""
    return str(
        article.get("data", {}).get("id")
        or article.get("_id")
        or article.get("article_id")
        or article.get("mongo_id")
        or ""
    )


def load_existing_article_ids(cur) -> set:
    cur.execute("""
        SELECT DISTINCT article_id
        FROM news_embeddings
        WHERE article_id IS NOT NULL AND article_id <> ''
    """)
    return {row[0] for row in cur.fetchall()}


def insert_embedding(cur, article_id, article, chunk, emb):
    cur.execute(
        """
        INSERT INTO news_embeddings
        (article_id, title, content, topic, sentiment_label,
         sentiment_score, source, scraped_at, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            article_id,
            article.get("title"),
            chunk,
            article.get("topic"),
            article.get("sentiment", {}).get("label"),
            article.get("sentiment", {}).get("score"),
            article.get("source"),
            article.get("scraped_at"),
            list(map(float, emb)),
        ),
    )


# -----------------
# Main loop implementation
# -----------------

# Get document count for tqdm total and diagnostic
try:
    total = news_coll.count_documents({})
    print("Document count in collection:", total)
except Exception as e:
    print("Warning: could not get document count:", str(e))
    total = None

if total == 0:
    print("No documents found in the configured Mongo collection. Check MONGO_URI, MONGO_DB, and MONGO_COLL.")
    # show a sample of databases/collections to help debug
    try:
        dbs = mongo.list_database_names()
        print("Databases:", dbs)
        for dbn in dbs:
            cols = mongo[dbn].list_collection_names()
            if cols:
                print("Sample collection: {}.{}".format(dbn, cols[0]))
                sample = mongo[dbn][cols[0]].find_one()
                print("Sample document:", sample)
                break
    except Exception as e:
        print("Unable to list databases/collections:", str(e))
    sys.exit(0)

# If pg is None we still want to run but skip writes (or you can exit instead)
if pg is None:
    print("Warning: Postgres not available. Script will attempt to compute embeddings but will not persist them.")

cur = None
if pg:
    cur = pg.cursor()
    existing_ids = load_existing_article_ids(cur)
    print("Articles already in Postgres:", len(existing_ids))
else:
    existing_ids = set()

# Add counters for diagnostics
_inserted_count = 0
_failed_inserts = 0
_processed_articles = 0

try:
    # find returns a cursor; provide total to tqdm for proper progress
    articles = news_coll.find({})
    # tqdm will show accurate progress only if total is provided
    for article in tqdm(articles, desc="Embedding articles", total=total):
        _processed_articles += 1
        # print one-liner to help debugging during iteration
        aid = resolve_article_id(article)
        if not aid:
            print("Skipping article without id field")
            continue
        if aid in existing_ids:
            continue
        text = article.get("text")
        if not text:
            continue
        chunks = chunk_text(text)
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embs = model.encode(batch, normalize_embeddings=True)
            for chunk, emb in zip(batch, embs):
                if cur:
                    try:
                        insert_embedding(cur, aid, article, chunk, emb)
                        _inserted_count += 1
                    except Exception as e:
                        _failed_inserts += 1
                        print("Insert error for article {}: {}".format(aid, e), file=sys.stderr)
                        # print full traceback for debugging
                        traceback.print_exc()
                        # show a small preview of the failing payload
                        try:
                            preview = (chunk[:120] + '...') if len(chunk) > 120 else chunk
                        except Exception:
                            preview = '<unable to get chunk preview>'
                        print('Payload preview (len):', len(chunk) if isinstance(chunk, str) else 'N/A', file=sys.stderr)
                        print('Chunk preview:', preview, file=sys.stderr)
                else:
                    # if no postgres, just show one example and continue
                    print("Computed embedding for chunk (len):", len(chunk))
finally:
    if cur:
        cur.close()

# Print final summary
print("Embedding run summary:")
print("  Articles processed:", _processed_articles)
print("  Embedding chunks inserted:", _inserted_count)
print("  Failed inserts:", _failed_inserts)
