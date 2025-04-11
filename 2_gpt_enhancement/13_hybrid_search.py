#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3G: Hybrid Search Verification

Goal: Perform both vector similarity search and full-text keyword search
using a sample query against the 'textbook_chunks' table. Uses the
OAuth/SSL connection method from script 10 for embedding generation.
Compares the results to identify overlapping chunks.

Requires:
- `psycopg2` or `psycopg2-binary` library.
- `openai` library (`pip install openai`).
- `requests` library (`pip install requests`).
- `pgvector` extension enabled in PostgreSQL (`CREATE EXTENSION IF NOT EXISTS vector;`).
- Environment variables for OAuth/SSL configured as used by script 10
  (RBC_OAUTH_CLIENT_ID, RBC_OAUTH_CLIENT_SECRET, RBC_SSL_SOURCE_PATH).
"""

import os
import sys
import json
import time
import traceback
import requests # Needed for OAuth
from pathlib import Path
import psycopg2
import psycopg2.extras # For DictCursor
from openai import OpenAI, APIError
from pgvector.psycopg2 import register_vector

# --- Configuration ---
# Source document ID to search within (optional, set to None to search all)
DOCUMENT_ID_TO_SEARCH = "ey_international_gaap_2024"
# Sample query string
QUERY_STRING = "Explain the accounting treatment for leases under IFRS 16"
# Number of results to retrieve for each search type
TOP_K = 20

# --- Embedding Configuration ---
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 2000

# --- Database Configuration (Self-contained) ---
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "iris_dev",
    "password": "",  # No password needed for local development
}

# --- API Connection Configuration (Copied from Script 10) ---
# Ensure these match your environment/script 10 setup
BASE_URL = os.environ.get("RBC_LLM_ENDPOINT", "https://api.example.com/v1") # Use env var or placeholder
OAUTH_URL = os.environ.get("RBC_OAUTH_URL", "https://api.example.com/oauth/token") # Use env var or placeholder
CLIENT_ID = os.environ.get("RBC_OAUTH_CLIENT_ID", "your_client_id") # Placeholder
CLIENT_SECRET = os.environ.get("RBC_OAUTH_CLIENT_SECRET", "your_client_secret") # Placeholder
SSL_SOURCE_PATH = os.environ.get("RBC_SSL_SOURCE_PATH", "/path/to/your/rbc-ca-bundle.cer") # Placeholder
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer" # Temporary local path for cert

# --- Internal Constants (Copied/Adapted from Script 10) ---
_SSL_CONFIGURED = False # Flag to avoid redundant SSL setup
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 5 # seconds

print("--- Starting Hybrid Search ---")

# --- Helper Functions (Database Connection) ---

def connect_to_db(params):
    """Connects to the PostgreSQL database."""
    conn = None
    try:
        print(f"Connecting to database '{params['dbname']}' on {params['host']}...")
        conn = psycopg2.connect(**params)
        # Register pgvector type handler
        register_vector(conn)
        print("Connection successful and pgvector registered.")
        return conn
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- Helper Functions (Copied/Adapted from Script 10 for OpenAI Connection) ---

def _setup_ssl(source_path=SSL_SOURCE_PATH, local_path=SSL_LOCAL_PATH):
    """Copies SSL cert locally and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True # Already configured

    print("Setting up SSL certificate...")
    try:
        source = Path(source_path)
        local = Path(local_path)

        if not source.is_file():
            print(f"ERROR: SSL source certificate not found at {source_path}", file=sys.stderr)
            return False

        local.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as source_file:
            content = source_file.read()
        with open(local, "wb") as dest_file:
            dest_file.write(content)

        os.environ["SSL_CERT_FILE"] = str(local)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local)
        print(f"SSL certificate configured successfully at: {local}")
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        print(f"ERROR: Error setting up SSL certificate: {e}", file=sys.stderr)
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH):
    """Retrieves OAuth token from the specified endpoint."""
    print("Attempting to get OAuth token...")
    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    try:
        # Ensure SSL is set up before making the request
        if not _SSL_CONFIGURED:
             print("ERROR: SSL not configured, cannot get OAuth token.", file=sys.stderr)
             return None

        response = requests.post(
            oauth_url,
            data=payload,
            timeout=30,
            verify=ssl_verify_path # Use the configured local path
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        token_data = response.json()
        oauth_token = token_data.get('access_token')
        if not oauth_token:
            print("ERROR: 'access_token' not found in OAuth response.", file=sys.stderr)
            return None
        print("OAuth token obtained successfully.")
        return oauth_token
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error getting OAuth token: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during OAuth token retrieval: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def create_openai_client(base_url=BASE_URL):
    """Sets up SSL, gets OAuth token, and creates the OpenAI client."""
    if not _setup_ssl():
        print("ERROR: Aborting client creation due to SSL setup failure.", file=sys.stderr)
        return None # SSL setup failed

    api_key = _get_oauth_token()
    if not api_key:
        print("ERROR: Aborting client creation due to OAuth token failure.", file=sys.stderr)
        return None # Token retrieval failed

    try:
        # Pass http_client using httpx if needed for custom SSL context,
        # otherwise rely on REQUESTS_CA_BUNDLE env var set in _setup_ssl
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
            # Example using httpx for custom SSL context if needed:
            # http_client=httpx.Client(verify=SSL_LOCAL_PATH)
        )
        print("OpenAI client created successfully using OAuth token.")
        return client
    except Exception as e:
        print(f"ERROR: Error creating OpenAI client: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

def generate_query_embedding(client, query: str, model: str, dimensions: int) -> list[float] | None:
    """Generates embedding for the query string using the provided client."""
    if not client:
        print("ERROR: OpenAI client not available for embedding generation.", file=sys.stderr)
        return None
    print(f"Generating embedding for query: '{query}'...")
    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = client.embeddings.create(
                input=[query], # API expects a list
                model=model,
                dimensions=dimensions
            )
            if response.data and response.data[0].embedding:
                print("Embedding generated successfully.")
                return response.data[0].embedding
            else:
                print(f"ERROR: No embedding data received from API (Attempt {attempt + 1}).", file=sys.stderr)
                last_exception = ValueError("No embedding data in API response")
                time.sleep(_RETRY_DELAY) # Wait before retry

        except APIError as e:
            print(f"WARNING: OpenAI API error during embedding generation (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY * (attempt + 1)) # Exponential backoff might be better
        except Exception as e:
            print(f"WARNING: Unexpected error during embedding generation (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY)

    print(f"ERROR: Failed to generate embedding after {_RETRY_ATTEMPTS} attempts.", file=sys.stderr)
    if last_exception:
        print(f"Last error: {last_exception}", file=sys.stderr)
    return None


# --- Search Functions ---

def perform_vector_search(cursor, query_embedding: list[float], top_k: int, doc_id: str | None):
    """Performs vector similarity search using cosine distance."""
    print(f"\n--- Performing Vector Similarity Search (Top {top_k}) ---")
    results = []
    if query_embedding is None:
        print("ERROR: Cannot perform vector search without query embedding.", file=sys.stderr)
        return results

    try:
        sql = """
            SELECT
                id,
                sequence_number,
                content,
                1 - (embedding <=> %s::vector) AS similarity_score -- Cosine similarity
            FROM textbook_chunks
        """
        params = [query_embedding]

        if doc_id:
            sql += " WHERE document_id = %s"
            params.append(doc_id)

        sql += " ORDER BY similarity_score DESC LIMIT %s;"
        params.append(top_k)

        cursor.execute(sql, params)
        results = cursor.fetchall()
        print(f"Found {len(results)} results via vector search.")

    except Exception as e:
        print(f"ERROR: Vector search failed: {e}", file=sys.stderr)
        if "operator does not exist" in str(e):
             print("HINT: Ensure the 'pgvector' extension is installed and enabled in your database (`CREATE EXTENSION IF NOT EXISTS vector;`).", file=sys.stderr)
        traceback.print_exc()

    return results

def perform_keyword_search(cursor, query: str, top_k: int, doc_id: str | None):
    """Performs full-text keyword search."""
    print(f"\n--- Performing Keyword Search (Top {top_k}) ---")
    results = []
    try:
        # Check if websearch_to_tsquery is available (PostgreSQL 11+)
        cursor.execute("SELECT 1 FROM pg_proc WHERE proname = 'websearch_to_tsquery'")
        has_websearch = cursor.fetchone() is not None
        
        # Use websearch_to_tsquery if available, fall back to to_tsquery
        # websearch_to_tsquery handles phrases in quotes, OR/AND operators, and - for negation
        if has_websearch:
            print("Using websearch_to_tsquery for more flexible search...")
            tsquery_func = "websearch_to_tsquery"
        else:
            print("Using to_tsquery with OR logic for more matches...")
            # Convert query to OR-based search by joining terms with |
            words = query.strip().split()
            query = ' | '.join(words)
            tsquery_func = "to_tsquery"
            
        # Enhanced ranking with normalization and weights
        sql = f"""
            SELECT
                id,
                sequence_number,
                content,
                ts_rank_cd(text_search_vector, {tsquery_func}('english', %s), 32) * 10 AS rank
            FROM textbook_chunks
            WHERE text_search_vector @@ {tsquery_func}('english', %s)
        """
        params = [query, query]

        if doc_id:
            sql += " AND document_id = %s"
            params.append(doc_id)

        sql += " ORDER BY rank DESC LIMIT %s;"
        params.append(top_k)

        cursor.execute(sql, params)
        results = cursor.fetchall()
        print(f"Found {len(results)} results via keyword search.")

    except Exception as e:
        print(f"ERROR: Keyword search failed: {e}", file=sys.stderr)
        traceback.print_exc()

    return results

def display_results(search_type: str, results: list, score_field: str):
    """Displays search results."""
    print(f"\n--- Top {len(results)} {search_type} Results ---")
    if not results:
        print("No results found.")
        return

    for i, record in enumerate(results):
        print(f"{i+1}. ID: {record['id']}, Seq: {record['sequence_number']}, Score/Rank: {record[score_field]:.4f}")
        print(f"   Content: {str(record['content'])[:200]}...") # Show more preview

# --- Main Execution Function ---
def check_text_search_setup(cursor):
    """Verifies the text_search_vector column setup in the database."""
    print("\n--- Checking Text Search Setup ---")
    try:
        # Check if text_search_vector column exists
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'textbook_chunks' AND column_name = 'text_search_vector';
        """)
        column_info = cursor.fetchone()
        
        if not column_info:
            print("WARNING: text_search_vector column not found in textbook_chunks table!")
            print("To create it, run: ALTER TABLE textbook_chunks ADD COLUMN text_search_vector tsvector;")
            return False
            
        print(f"✅ Found text_search_vector column (type: {column_info[1]})")
        
        # Check if the column has data
        cursor.execute("SELECT COUNT(*) FROM textbook_chunks WHERE text_search_vector IS NOT NULL;")
        count = cursor.fetchone()[0]
        print(f"✅ Found {count} rows with non-NULL text_search_vector values")
        
        if count == 0:
            print("WARNING: No text search vectors found. Consider updating with:")
            print("UPDATE textbook_chunks SET text_search_vector = to_tsvector('english', content);")
        
        # Check for index
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'textbook_chunks' AND indexdef LIKE '%text_search_vector%';
        """)
        index = cursor.fetchone()
        
        if not index:
            print("WARNING: No index found on text_search_vector. Consider creating one with:")
            print("CREATE INDEX idx_textbook_chunks_tsv ON textbook_chunks USING GIN(text_search_vector);")
        else:
            print(f"✅ Found index on text_search_vector: {index[0]}")
            
        return True
        
    except Exception as e:
        print(f"ERROR checking text search setup: {e}")
        return False

def perform_hybrid_search(query=QUERY_STRING):
    """Generates embedding, runs searches, compares results."""

    # 1. Create OpenAI Client (using OAuth/SSL method)
    client = create_openai_client()
    if not client:
        print("ERROR: Failed to create OpenAI client. Cannot generate query embedding.", file=sys.stderr)
        raise RuntimeError("OpenAI client initialization failed.")

    # 2. Generate Query Embedding
    query_embedding = generate_query_embedding(
        client=client,
        query=query,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS
    )
    if query_embedding is None:
        print("ERROR: Failed to generate query embedding. Aborting search.", file=sys.stderr)
        raise RuntimeError("Query embedding generation failed.")

    # 3. Connect to Database
    conn = connect_to_db(DB_PARAMS)
    if conn is None:
        print("Aborting due to database connection failure.", file=sys.stderr)
        raise RuntimeError("Database connection failed.")

    cursor = None
    vector_results = []
    keyword_results = []
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Check the text search setup first
        check_text_search_setup(cursor)

        # 4. Perform Searches
        vector_results = perform_vector_search(cursor, query_embedding, TOP_K, DOCUMENT_ID_TO_SEARCH)
        keyword_results = perform_keyword_search(cursor, query, TOP_K, DOCUMENT_ID_TO_SEARCH)

        # 5. Display Results
        display_results("Vector Similarity", vector_results, "similarity_score")
        display_results("Keyword", keyword_results, "rank")

        # 6. Compare Results
        print("\n--- Overlapping Results ---")
        vector_ids = {r['id'] for r in vector_results}
        keyword_ids = {r['id'] for r in keyword_results}
        overlap_ids = vector_ids.intersection(keyword_ids)

        if overlap_ids:
            print(f"Found {len(overlap_ids)} overlapping chunk IDs:")
            # Optionally fetch details for overlapping IDs
            overlap_details = []
            if overlap_ids:
                 # Create a placeholder string like %s,%s,%s
                 placeholders = ','.join(['%s'] * len(overlap_ids))
                 sql_overlap = f"""
                     SELECT id, sequence_number, content
                     FROM textbook_chunks
                     WHERE id IN ({placeholders})
                     ORDER BY sequence_number;
                 """
                 cursor.execute(sql_overlap, list(overlap_ids))
                 overlap_details = cursor.fetchall()

            for i, detail in enumerate(overlap_details):
                 print(f"  {i+1}. ID: {detail['id']}, Seq: {detail['sequence_number']}, Content: {str(detail['content'])[:100]}...")
        else:
            print("No overlapping results found between the two search methods.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"ERROR: Hybrid search failed: {error}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n--- Hybrid Search Finished ---")


# --- To run in a notebook cell, call the function: ---
# perform_hybrid_search(query="Your query here")
# or just:
# perform_hybrid_search() # Uses default query

# --- Or run as a script ---
if __name__ == "__main__":
    try:
        # You could add argparse here to take query from command line
        perform_hybrid_search()
    except Exception as e:
        print(f"FATAL ERROR during hybrid search: {e}", file=sys.stderr)
        sys.exit(1)
