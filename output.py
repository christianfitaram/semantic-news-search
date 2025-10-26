import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Allow passing the URI as first arg for convenience
uri = None
if len(sys.argv) > 1:
    uri = sys.argv[1]
else:
    uri = os.getenv("POSTGRES_URI")

if not uri:
    print("Error: POSTGRES_URI not set. Set the environment variable or pass the connection string as the first argument.")
    sys.exit(1)

conn = None
cur = None
try:
    conn = psycopg2.connect(uri)
    cur = conn.cursor()

    # Check table existence (Postgres specific)
    cur.execute("SELECT to_regclass('public.news_embeddings')")
    exists = cur.fetchone()[0]
    if not exists:
        print("Table 'news_embeddings' does not exist in this database/schema.")
        # show available tables in public schema for debugging
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename LIMIT 50")
        tables = cur.fetchall()
        print('Public tables (sample):', [t[0] for t in tables])
    else:
        cur.execute("SELECT COUNT(*) FROM news_embeddings;")
        count = cur.fetchone()[0]
        print("news_embeddings row count:", count)

        # Print a small sample of recent rows if any
        if count > 0:
            try:
                cur.execute("SELECT article_id, title, scraped_at FROM news_embeddings ORDER BY scraped_at DESC NULLS LAST LIMIT 5")
                rows = cur.fetchall()
                print("Sample rows:")
                for r in rows:
                    print(r)
            except Exception:
                # Fall back to a generic sample if scraped_at doesn't exist
                cur.execute("SELECT * FROM news_embeddings LIMIT 5")
                rows = cur.fetchall()
                print("Sample rows (fallback):")
                for r in rows:
                    print(r)

except Exception as e:
    print("Error connecting/querying Postgres:", e)
finally:
    if cur:
        try:
            cur.close()
        except Exception:
            pass
    if conn:
        try:
            conn.close()
        except Exception:
            pass
