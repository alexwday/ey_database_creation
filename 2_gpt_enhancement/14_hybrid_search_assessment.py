#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3H: Response Generation from Retrieved Chunks

Goal: Format retrieved chunks as "cards" and provide them to GPT 
to generate a comprehensive response that cites specific chapters and sections.

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
# Number of results to retrieve for search
TOP_K = 10
# API model for response generation
RESPONSE_MODEL = "gpt-4o"
# Maximum tokens for the response
MAX_RESPONSE_TOKENS = 4000
# Temperature for response generation
RESPONSE_TEMPERATURE = 0.7

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

def perform_hybrid_search(cursor, query: str, query_embedding: list[float], top_k: int, doc_id: str | None):
    """
    Performs hybrid search combining vector similarity and keyword search.
    Returns a combined ranked list of results.
    """
    print(f"\n--- Performing Hybrid Search (Top {top_k}) ---")
    results = []
    
    if query_embedding is None:
        print("ERROR: Cannot perform vector component of hybrid search without embedding.", file=sys.stderr)
        return results

    try:
        # Check if websearch_to_tsquery is available (PostgreSQL 11+)
        cursor.execute("SELECT 1 FROM pg_proc WHERE proname = 'websearch_to_tsquery'")
        has_websearch = cursor.fetchone() is not None
        
        # Use websearch_to_tsquery if available, fall back to to_tsquery
        if has_websearch:
            print("Using websearch_to_tsquery for keyword component...")
            tsquery_func = "websearch_to_tsquery"
            # Keep original query for websearch_to_tsquery as it handles operators
            search_query = query
        else:
            print("Using to_tsquery for keyword component...")
            tsquery_func = "to_tsquery"
            # Extract words, filter out very short ones
            words = [w for w in query.strip().split() if len(w) > 2]
            if not words:
                words = query.strip().split()  # Use all words if filtering removed everything
            # Create OR-based query to find documents with ANY of the keywords
            search_query = ' | '.join(words)
            print(f"Using OR-based query: {search_query}")

        # Hybrid search SQL combining vector and keyword components
        # Weighting: 0.7 for vector similarity, 0.3 for text search relevance
        # This ratio can be adjusted based on your requirements
        sql = f"""
            WITH vector_results AS (
                SELECT 
                    id,
                    1 - (embedding <=> %s::vector) AS vector_score
                FROM textbook_chunks
                WHERE 1=1
                {" AND document_id = %s" if doc_id else ""}
            ),
            text_results AS (
                SELECT 
                    id,
                    ts_rank_cd(text_search_vector, {tsquery_func}('english', %s), 2|32) * 10 AS text_score
                FROM textbook_chunks
                WHERE text_search_vector @@ {tsquery_func}('english', %s)
                {" AND document_id = %s" if doc_id else ""}
            ),
            combined_results AS (
                SELECT 
                    c.id,
                    c.sequence_number,
                    c.document_id,
                    c.chapter_name,
                    c.section_hierarchy,
                    c.section_title,
                    c.content,
                    c.standard,
                    c.standard_codes,
                    c.tags,
                    c.page_start,
                    c.page_end, 
                    c.importance_score,
                    COALESCE(v.vector_score, 0) AS vector_score,
                    COALESCE(t.text_score, 0) AS text_score,
                    (COALESCE(v.vector_score, 0) * 0.7) + (COALESCE(t.text_score, 0) * 0.3) AS combined_score
                FROM textbook_chunks c
                LEFT JOIN vector_results v ON c.id = v.id
                LEFT JOIN text_results t ON c.id = t.id
                WHERE COALESCE(v.vector_score, 0) > 0 OR COALESCE(t.text_score, 0) > 0
            )
            SELECT * FROM combined_results
            ORDER BY combined_score DESC
            LIMIT %s;
        """
        
        # Prepare parameters
        params = [query_embedding, search_query, search_query, top_k]
        if doc_id:
            # Insert document_id parameter for both vector and text queries
            params = [query_embedding, doc_id, search_query, search_query, doc_id, top_k]
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        print(f"Found {len(results)} results via hybrid search.")

    except Exception as e:
        print(f"ERROR: Hybrid search failed: {e}", file=sys.stderr)
        traceback.print_exc()

    return results

# --- Response Generation Functions ---

def format_chunks_as_cards(results):
    """
    Formats database results into "cards" for better GPT context understanding.
    
    Args:
        results: List of database result rows (as dict cursors)
    
    Returns:
        Formatted string with all chunks as cards
    """
    cards = []
    
    for i, record in enumerate(results):
        card_parts = [f"--- CARD {i+1} ---"]
        
        # Card metadata
        chapter_name = record.get('chapter_name', 'Unknown Chapter')
        # Extract chapter number from section_hierarchy if available
        section_hierarchy = record.get('section_hierarchy', '')
        section_title = record.get('section_title', 'Unknown Section')
        
        # Format chapter and section info
        card_parts.append(f"Chapter: {chapter_name}")
        card_parts.append(f"Section: {section_title}")
        if section_hierarchy:
            card_parts.append(f"Section Hierarchy: {section_hierarchy}")
        
        # Add page range if available
        page_start = record.get('page_start')
        page_end = record.get('page_end')
        if page_start and page_end:
            card_parts.append(f"Pages: {page_start}-{page_end}")
        
        # Add importance score if available
        importance_score = record.get('importance_score')
        if importance_score is not None:
            card_parts.append(f"Importance Score: {importance_score:.2f}")
        
        # Add standard information if available
        standard = record.get('standard')
        if standard:
            card_parts.append(f"Standard: {standard}")
        
        standard_codes = record.get('standard_codes')
        if standard_codes and isinstance(standard_codes, list) and standard_codes:
            card_parts.append(f"Standard Codes: {', '.join(standard_codes)}")
        
        tags = record.get('tags')
        if tags and isinstance(tags, list) and tags:
            card_parts.append(f"Tags: {', '.join(tags)}")
        
        # Add search scores for reference
        vector_score = record.get('vector_score', 0)
        text_score = record.get('text_score', 0)
        combined_score = record.get('combined_score', 0)
        card_parts.append(f"Relevance Scores: Vector={vector_score:.4f}, Text={text_score:.4f}, Combined={combined_score:.4f}")
        
        # Add content
        content = record.get('content', '')
        card_parts.append("\nCONTENT:")
        card_parts.append(content)
        
        cards.append("\n".join(card_parts))
    
    # Join all cards with clear separation
    return "\n\n" + "\n\n".join(cards) + "\n\n"

def generate_response_from_chunks(client, query, formatted_chunks):
    """
    Generates a GPT response based on the query and formatted chunks.
    
    Args:
        client: OpenAI client
        query: User's query/question
        formatted_chunks: Chunks formatted as cards
    
    Returns:
        Generated response from GPT
    """
    print("\n--- Generating Response from Retrieved Chunks ---")
    
    system_message = """You are a specialized accounting research assistant with expertise in IFRS and US GAAP standards.
Your task is to answer accounting questions based ONLY on the information provided in the context cards.
You must NEVER use information from your training data - rely exclusively on the cards provided.

When answering:
1. Be thorough, accurate and precise, focusing only on accounting/financial reporting relevance.
2. You MUST cite specific chapters and sections that contain your information using the format: [Chapter X, Section Y] for each point.
3. If multiple cards provide supporting information for a point, cite all relevant sources.
4. Do not fabricate information - if the provided cards don't contain sufficient information, acknowledge the limitations.
5. Structure your response with clear headings if appropriate.
6. Provide a concise summary at the end (2-3 sentences).

Remember: Every significant point you make MUST include chapter and section references."""

    user_message = f"""Question: {query}

Below are the reference cards that contain information to answer this question. Use ONLY this information, not your training data.

{formatted_chunks}

Please provide a comprehensive answer with proper chapter and section citations for each point."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Make API call
        last_exception = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                print(f"Making API call for response generation (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...")
                
                response = client.chat.completions.create(
                    model=RESPONSE_MODEL,
                    messages=messages,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    temperature=RESPONSE_TEMPERATURE,
                    stream=False
                )
                
                print("API call for response generation successful.")
                response_content = response.choices[0].message.content
                
                # Print token usage information
                usage_info = response.usage
                if usage_info:
                    print(f"Token Usage - Prompt: {usage_info.prompt_tokens}, Completion: {usage_info.completion_tokens}, Total: {usage_info.total_tokens}")
                
                return response_content
                
            except APIError as e:
                print(f"API Error on attempt {attempt + 1}: {e}", file=sys.stderr)
                last_exception = e
                time.sleep(_RETRY_DELAY * (attempt + 1))
            except Exception as e:
                print(f"Non-API Error on attempt {attempt + 1}: {e}", file=sys.stderr)
                last_exception = e
                time.sleep(_RETRY_DELAY)
        
        # If we get here, all attempts failed
        error_msg = f"ERROR: Failed to generate response after {_RETRY_ATTEMPTS} attempts."
        if last_exception:
            error_msg += f" Last error: {str(last_exception)}"
        print(error_msg, file=sys.stderr)
        return f"Error generating response: {error_msg}"
    
    except Exception as e:
        print(f"ERROR: Unexpected error during response generation: {e}", file=sys.stderr)
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

# --- Main Function ---

def generate_response(query):
    """
    Main function to retrieve chunks and generate a response.
    Can be called directly from a Jupyter notebook cell.
    
    Args:
        query: User's query/question
    
    Returns:
        Generated response from GPT based on retrieved chunks
    """
    print(f"--- Starting Response Generation Process ---")
    print(f"Query: '{query}'")

    # 1. Create OpenAI Client (using OAuth/SSL method)
    client = create_openai_client()
    if not client:
        return "ERROR: Failed to create OpenAI client. Cannot proceed with response generation."

    # 2. Generate Query Embedding
    query_embedding = generate_query_embedding(
        client=client,
        query=query,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS
    )
    if query_embedding is None:
        return "ERROR: Failed to generate query embedding. Cannot perform vector search component."

    # 3. Connect to Database
    conn = connect_to_db(DB_PARAMS)
    if conn is None:
        return "ERROR: Database connection failed. Cannot retrieve chunks for response generation."

    response_text = None
    cursor = None
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 4. Perform Hybrid Search to retrieve relevant chunks
        search_results = perform_hybrid_search(
            cursor=cursor,
            query=query,
            query_embedding=query_embedding,
            top_k=TOP_K,
            doc_id=DOCUMENT_ID_TO_SEARCH
        )
        
        if not search_results:
            return "No relevant information found for your query in the database."
        
        # 5. Format chunks as cards
        formatted_chunks = format_chunks_as_cards(search_results)
        
        # 6. Generate response from chunks
        response_text = generate_response_from_chunks(
            client=client,
            query=query,
            formatted_chunks=formatted_chunks
        )

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"ERROR: Response generation process failed: {error}", file=sys.stderr)
        traceback.print_exc()
        response_text = f"Error during response generation process: {str(error)}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n--- Response Generation Process Finished ---")
    return response_text

# Example of how to use in a Jupyter notebook cell:
# response = generate_response("Explain the accounting treatment for leases under IFRS 16")
# print(response)