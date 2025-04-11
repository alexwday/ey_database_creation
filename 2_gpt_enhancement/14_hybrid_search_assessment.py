#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3H: Hybrid Search Assessment

Goal: Evaluate retrieval quality for hybrid search by sending chunks to GPT
for relevance assessment. Uses OAuth/SSL connection method for API calls.
Compares vector and keyword search results with a relevance score from GPT.

For use in Jupyter notebook cells.

Requires:
- `psycopg2` or `psycopg2-binary` library.
- `openai` library (`pip install openai`).
- `requests` library (`pip install requests`).
- `pgvector` extension enabled in PostgreSQL.
- Environment variables for OAuth/SSL configured.
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
# Number of results to retrieve for each search type
TOP_K = 3
# API model for relevance judgments
JUDGE_MODEL = "gpt-4o"

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
                chapter_name,
                section_title,
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
        # Extract words, filter out very short ones that could be noise
        words = [w for w in query.strip().split() if len(w) > 2]
        
        if not words:
            print("Warning: Query contains no usable keywords after filtering.")
            words = query.strip().split()  # Use all words if filtering removed everything
            
        # Create OR-based query to find documents with ANY of the keywords
        or_query = ' | '.join(words)
        print(f"Searching for documents containing any of these terms: {or_query}")
            
        # Check if websearch_to_tsquery is available (PostgreSQL 11+)
        cursor.execute("SELECT 1 FROM pg_proc WHERE proname = 'websearch_to_tsquery'")
        has_websearch = cursor.fetchone() is not None
        
        # Use websearch_to_tsquery if available, fall back to to_tsquery
        if has_websearch:
            print("Using websearch_to_tsquery for more flexible search...")
            tsquery_func = "websearch_to_tsquery"
            # Keep original query for websearch_to_tsquery as it handles operators
            search_query = query
        else:
            print("Using to_tsquery with OR logic...")
            tsquery_func = "to_tsquery"
            # Use our OR-based query for to_tsquery
            search_query = or_query
            
        # Enhanced ranking with combined normalization:
        # - 2: Normalize by document length (fair comparison between short/long chunks)
        # - 32: Compress scores between 0-1 to avoid extreme values
        # Multiply by 10 for more readable scores
        sql = f"""
            SELECT
                id,
                sequence_number,
                content,
                chapter_name,
                section_title,
                ts_rank_cd(text_search_vector, {tsquery_func}('english', %s), 2|32) * 10 AS rank,
                LENGTH(content) AS content_length
            FROM textbook_chunks
            WHERE text_search_vector @@ {tsquery_func}('english', %s)
        """
        params = [search_query, search_query]

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

def evaluate_chunk_relevance(client, query, chunk_content, chunk_info):
    """
    Send chunk to GPT for relevance assessment.
    Returns structured assessment data.
    """
    system_message = """
    You are an expert system evaluating search retrieval quality. 
    Your task is to judge how relevant a retrieved text chunk is to a search query.
    Give an honest, critical assessment of whether this chunk actually answers the query.
    
    Rate relevance on a scale of 1-10 where:
    - 1-3: Not relevant or marginally relevant
    - 4-6: Somewhat relevant but incomplete
    - 7-10: Very relevant, directly answers the query
    
    Provide your assessment in JSON format with these fields:
    - relevance_score: Numeric score (1-10)
    - analysis: Brief explanation of your scoring (1-2 sentences)
    - key_points: List of 1-3 key points from the chunk relevant to the query
    - missing_info: What important information is missing (if any)
    """
    
    user_message = f"""
    Query: {query}
    
    Retrieved Chunk Information:
    ID: {chunk_info.get('id')}
    Chapter: {chunk_info.get('chapter_name', 'N/A')}
    Section: {chunk_info.get('section_title', 'N/A')}
    
    Chunk Content:
    {chunk_content}
    
    Provide your assessment in JSON format.
    """
    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )
        
        # Extract and parse JSON response
        response_text = response.choices[0].message.content
        return json.loads(response_text)
    
    except Exception as e:
        print(f"ERROR: Chunk evaluation failed: {e}", file=sys.stderr)
        return {
            "relevance_score": 0,
            "analysis": f"Error evaluating chunk: {str(e)[:100]}",
            "key_points": [],
            "missing_info": "Evaluation failed"
        }

def assess_search_results(client, query, results, search_type):
    """Assess all results from a search method."""
    print(f"\n--- Assessing {search_type} Results ---")
    
    assessed_results = []
    
    for i, record in enumerate(results):
        print(f"\nEvaluating {search_type} result #{i+1} (ID: {record['id']})...")
        
        # Prepare info for assessment
        chunk_info = {
            "id": record['id'],
            "chapter_name": record.get('chapter_name', 'Unknown Chapter'),
            "section_title": record.get('section_title', 'Unknown Section'),
            "sequence_number": record['sequence_number']
        }
        
        # Get score based on search type
        if search_type == "Vector":
            search_score = float(record['similarity_score'])
            score_label = "similarity_score"
        else:
            search_score = float(record['rank'])
            score_label = "rank"
        
        # Evaluate chunk
        assessment = evaluate_chunk_relevance(client, query, record['content'], chunk_info)
        
        # Store results
        result_data = {
            "chunk_info": chunk_info,
            "search_score": search_score,
            "score_label": score_label,
            "assessment": assessment,
            "content_preview": str(record['content'])[:150] + "..."
        }
        
        assessed_results.append(result_data)
        
        # Display assessment summary
        print(f"  Search Score ({score_label}): {search_score:.6f}")
        print(f"  Relevance Score: {assessment['relevance_score']}/10")
        print(f"  Analysis: {assessment['analysis']}")
    
    return assessed_results

def display_assessment_results(assessed_results, search_type):
    """Display assessment results in a clean format."""
    print(f"\n{'='*80}")
    print(f"ASSESSMENT SUMMARY: {search_type.upper()} SEARCH")
    print(f"{'='*80}")
    
    for i, result in enumerate(assessed_results):
        chunk_info = result["chunk_info"]
        assessment = result["assessment"]
        
        print(f"\n{search_type} Result #{i+1}:")
        print(f"  ID: {chunk_info['id']}, Seq: {chunk_info['sequence_number']}")
        print(f"  Location: {chunk_info.get('chapter_name', 'Unknown')} > {chunk_info.get('section_title', 'Unknown')}")
        print(f"  {search_type} Score: {result['search_score']:.6f}")
        print(f"  Relevance Score: {assessment['relevance_score']}/10")
        print(f"  Analysis: {assessment['analysis']}")
        
        if assessment.get('key_points'):
            print("  Key Points:")
            for point in assessment['key_points']:
                print(f"    • {point}")
                
        if assessment.get('missing_info') and assessment['missing_info'] != "None":
            print(f"  Missing Information: {assessment['missing_info']}")
            
        print(f"  Preview: {result['content_preview']}")
    
    # Calculate average relevance
    avg_relevance = sum(r['assessment']['relevance_score'] for r in assessed_results) / len(assessed_results) if assessed_results else 0
    print(f"\nAverage Relevance Score: {avg_relevance:.2f}/10")

def evaluate_hybrid_search(query):
    """
    Main function to run search, assess relevance, and display results.
    Can be called directly from a Jupyter notebook cell.
    """
    print(f"--- Starting Hybrid Search Assessment ---")
    print(f"Query: '{query}'")

    # 1. Create OpenAI Client (using OAuth/SSL method)
    client = create_openai_client()
    if not client:
        print("ERROR: Failed to create OpenAI client. Cannot proceed.", file=sys.stderr)
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
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 4. Perform Searches
        vector_results = perform_vector_search(cursor, query_embedding, TOP_K, DOCUMENT_ID_TO_SEARCH)
        keyword_results = perform_keyword_search(cursor, query, TOP_K, DOCUMENT_ID_TO_SEARCH)

        # 5. Assess Relevance of Search Results
        vector_assessments = assess_search_results(client, query, vector_results, "Vector")
        keyword_assessments = assess_search_results(client, query, keyword_results, "Keyword")
        
        # 6. Display Assessment Results
        display_assessment_results(vector_assessments, "Vector")
        display_assessment_results(keyword_assessments, "Keyword")
        
        # 7. Compare Overall Performance
        vector_avg = sum(r['assessment']['relevance_score'] for r in vector_assessments) / len(vector_assessments) if vector_assessments else 0
        keyword_avg = sum(r['assessment']['relevance_score'] for r in keyword_assessments) / len(keyword_assessments) if keyword_assessments else 0
        
        print("\n\n--- OVERALL COMPARISON ---")
        print(f"Vector Search Average Relevance: {vector_avg:.2f}/10")
        print(f"Keyword Search Average Relevance: {keyword_avg:.2f}/10")
        
        if vector_avg > keyword_avg:
            print(f"Vector search performed better by {vector_avg - keyword_avg:.2f} points")
        elif keyword_avg > vector_avg:
            print(f"Keyword search performed better by {keyword_avg - vector_avg:.2f} points")
        else:
            print("Both search methods performed equally")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"ERROR: Assessment failed: {error}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n--- Hybrid Search Assessment Finished ---")
    
    # Return the performance metrics for potential further analysis
    return {
        "vector_avg": vector_avg if 'vector_avg' in locals() else 0,
        "keyword_avg": keyword_avg if 'keyword_avg' in locals() else 0,
    }

# Example of how to use in a Jupyter notebook cell:
# evaluate_hybrid_search("Explain the accounting treatment for leases under IFRS 16")