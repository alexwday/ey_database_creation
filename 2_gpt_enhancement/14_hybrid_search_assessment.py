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
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pgvector.psycopg2 import register_vector
import json # For parsing GPT relevance response
import itertools # For grouping in gap filling
from tabulate import tabulate # For better table printing

# --- Configuration ---
# --- Search & Reranking Configuration ---
# Source document ID to search within (optional, set to None to search all)
DOCUMENT_ID_TO_SEARCH = "ey_international_gaap_2024"
# Number of results to retrieve initially (before filtering/expansion)
INITIAL_K = 20
# Final number of results to pass to GPT (after all steps) - Set to None to keep all
FINAL_K = None # Keep all results after processing
# API model for response generation
RESPONSE_MODEL = "gpt-4o"
# API model for summary relevance check (can be faster/cheaper if needed)
RELEVANCE_MODEL = "gpt-3.5-turbo"
# Importance factor for reranking
IMPORTANCE_FACTOR = 0.2
# Section expansion thresholds (page count)
SECTION_EXPANSION_TOP_K_THRESHOLD = 5 # Rank threshold for expanded page limit
SECTION_EXPANSION_GENERAL_THRESHOLD = 3 # Default page limit for expansion
SECTION_EXPANSION_EXPANDED_THRESHOLD = 6 # Expanded page limit for top K or multi-chunk sections
# Gap filling threshold (page count)
GAP_FILL_MAX_GAP = 3
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

def perform_hybrid_search(cursor, query: str, query_embedding: list[float], initial_k: int, doc_id: str | None):
    """
    Performs hybrid search combining vector similarity and keyword search.
    Returns a combined ranked list of results.
    """
    print(f"\n--- Performing Initial Hybrid Search (Retrieving Top {initial_k}) ---") # Updated log message
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
                    c.summary,
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
            LIMIT %s; -- Use initial_k for the limit
        """

        # Prepare parameters
        params = [query_embedding, search_query, search_query, initial_k] # Use initial_k
        if doc_id:
            # Insert document_id parameter for both vector and text queries
            params = [query_embedding, doc_id, search_query, search_query, doc_id, initial_k] # Use initial_k

        cursor.execute(sql, params)
        results = cursor.fetchall()
        print(f"Found {len(results)} results via hybrid search.")

    except Exception as e:
        print(f"ERROR: Hybrid search failed: {e}", file=sys.stderr)
        traceback.print_exc()

    return results


# --- Reranking & Filtering Functions ---

def filter_by_summary_relevance(client: OpenAI, query: str, results: list[dict], model: str = RELEVANCE_MODEL) -> list[dict]:
    """
    Uses GPT to classify chunk summaries as relevant (1) or irrelevant (0) to the query.
    Filters out irrelevant chunks.
    """
    print(f"\n--- Step 1: Filtering {len(results)} results by summary relevance using {model} ---")
    if not results:
        return []

    # Prepare summaries for GPT
    summaries_data = []
    for i, record in enumerate(results):
        # Use a unique identifier for each chunk in the prompt
        chunk_id = record.get('id')
        summary = record.get('summary', '')
        if chunk_id and summary:
            summaries_data.append({"id": chunk_id, "summary": summary})
        else:
            print(f"WARNING: Skipping result index {i} due to missing id or summary.", file=sys.stderr)

    if not summaries_data:
        print("WARNING: No valid summaries found to send for relevance check.", file=sys.stderr)
        return results # Return original results if none could be processed

    # Construct prompt for GPT
    prompt_summaries = "\n".join([f"ID: {item['id']}\nSummary: {item['summary']}\n---" for item in summaries_data])

    system_message = """You are an assistant tasked with evaluating the relevance of text summaries to a user's query.
Analyze the user's query and each provided summary.
For each summary ID, determine if the summary is:
- Directly relevant or highly related to the query (output 1)
- Completely irrelevant or unrelated to the query (output 0)

Respond ONLY with a valid JSON object where keys are the summary IDs (as strings) and values are either 1 (relevant) or 0 (irrelevant).
Example response format: {"chunk_id_1": 1, "chunk_id_2": 0, "chunk_id_3": 1}
Do not include any explanations or introductory text outside the JSON object."""

    user_message = f"""User Query: "{query}"

Evaluate the relevance of the following summaries to the query:
---
{prompt_summaries}
---
Provide your response as a single JSON object mapping each ID to 1 (relevant) or 0 (irrelevant)."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    relevance_map = {}
    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            print(f"Calling {model} for summary relevance check (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2, # Low temperature for classification
                response_format={"type": "json_object"} # Request JSON output
            )
            response_content = response.choices[0].message.content
            relevance_map = json.loads(response_content)
            # Validate format (simple check)
            if not isinstance(relevance_map, dict) or not all(isinstance(k, str) and v in [0, 1] for k, v in relevance_map.items()):
                 raise ValueError("Invalid JSON format received from relevance check API.")
            print("Summary relevance check successful.")
            break # Success
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to decode JSON response from relevance check (Attempt {attempt + 1}): {e}. Response: {response_content}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY)
        except (APIError, RateLimitError, APITimeoutError) as e:
            print(f"WARNING: API error during relevance check (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"WARNING: Unexpected error during relevance check (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY)

    if not relevance_map:
        print("ERROR: Failed to get relevance classifications after multiple attempts. Skipping filtering.", file=sys.stderr)
        if last_exception: print(f"  Last error: {last_exception}", file=sys.stderr)
        return results # Return original results if API failed

    # Filter results based on relevance map
    filtered_results = []
    removed_count = 0
    for record in results:
        chunk_id = record.get('id')
        # Default to relevant if ID wasn't processed or returned by GPT
        is_relevant = relevance_map.get(str(chunk_id), 1)
        if is_relevant == 1:
            filtered_results.append(record)
        else:
            removed_count += 1
            print(f"  - Filtering out chunk ID {chunk_id} (summary deemed irrelevant).")

    print(f"Finished summary filtering. Kept {len(filtered_results)} results, removed {removed_count}.")
    # Return both the filtered list and the map for logging
    return filtered_results, relevance_map


def expand_short_sections(cursor, results: list[dict], top_k_threshold: int, general_threshold: int, expanded_threshold: int) -> tuple[list[dict | list[dict]], set]:
    """
    Expands chunks belonging to short sections by fetching all chunks for that section.
    Returns a tuple containing:
    - The processed list (with groups for expanded sections).
    - A set of chunk IDs that were added during expansion (excluding the original triggering chunk).
    """
    print(f"\n--- Step 2: Expanding short sections (Gen Threshold: {general_threshold}pg, Exp Threshold: {expanded_threshold}pg for Top {top_k_threshold} or multi-chunk) ---")
    if not results: return [], set()

    processed_results = []
    expansion_log_data = []
    headers_expansion = ["Orig Chunk ID", "Rank", "Section Pages", "Threshold", "Action", "Added Chunks"]
    added_chunk_ids = set() # Track IDs added by this step

    # Identify sections already present more than once
    section_keys = set()
    multi_chunk_sections = set()
    for record in results:
        key = (record.get('document_id'), record.get('chapter_name'), record.get('section_hierarchy'))
        if key in section_keys:
            multi_chunk_sections.add(key)
        section_keys.add(key)

    # Keep track of sections we've already expanded to avoid redundant DB calls
    expanded_sections = set() # Store section_key tuples
    original_chunk_ids_in_results = {r.get('id') for r in results if isinstance(r, dict)} # IDs before expansion

    for record in results:
        orig_chunk_id = record.get('id')
        doc_id = record.get('document_id')
        chapter = record.get('chapter_name')
        hierarchy = record.get('section_hierarchy')
        page_start = record.get('page_start')
        page_end = record.get('page_end')
        rank = record.get('rank')
        section_key = (doc_id, chapter, hierarchy)

        log_row = [orig_chunk_id or "N/A", rank or "N/A", "N/A", "N/A", "Keep Single", 0] # Default log row

        # Skip if this section has already been fully added by a previous expansion
        if section_key in expanded_sections:
            # If the *original* chunk is being processed again after its section was expanded, skip it.
            # This prevents adding the single chunk back after the group was added.
            # We don't log this skip action as it's internal cleanup.
            continue

        # Determine if expansion is needed
        should_expand = False
        section_length = "N/A"
        threshold = "N/A"
        if page_start is not None and page_end is not None and rank is not None:
            section_length = page_end - page_start + 1
            log_row[2] = section_length
            # Determine threshold: expanded if rank is low OR section appears multiple times
            threshold = expanded_threshold if (rank <= top_k_threshold or section_key in multi_chunk_sections) else general_threshold
            log_row[3] = threshold

            if section_length <= threshold:
                should_expand = True
                # print(f"  - Section '{hierarchy}' (Rank {rank}, Len {section_length}pg) meets threshold ({threshold}pg). Checking for expansion.") # Replaced by table log
            # else:
                 # print(f"  - Section '{hierarchy}' (Rank {rank}, Len {section_length}pg) exceeds threshold ({threshold}pg). Keeping single chunk.") # Replaced by table log

        if should_expand:
            try:
                # print(f"    Fetching all chunks for section: Doc='{doc_id}', Chapter='{chapter}', Hierarchy='{hierarchy}'") # Replaced by table log
                sql = """
                    SELECT * FROM textbook_chunks
                    WHERE document_id = %s AND chapter_name = %s AND section_hierarchy = %s
                    ORDER BY sequence_number;
                """
                cursor.execute(sql, (doc_id, chapter, hierarchy))
                section_chunks_raw = cursor.fetchall()
                num_found = len(section_chunks_raw)

                if num_found > 1: # Only expand if there are actually more chunks in DB
                    section_chunks = [dict(chunk) for chunk in section_chunks_raw]
                    group_info = {
                        'type': 'group',
                        'original_rank': rank,
                        'original_combined_score': record.get('combined_score'),
                        'original_importance_score': record.get('importance_score'),
                        'chunks': section_chunks
                    }
                    processed_results.append(group_info)
                    expanded_sections.add(section_key) # Mark section as expanded
                    log_row[4] = f"Expand Group ({num_found} total)"
                    log_row[5] = num_found - 1 # Number of *added* chunks (total - original)
                    # Track newly added chunk IDs (excluding the original trigger chunk ID)
                    for chunk in section_chunks:
                        chunk_id = chunk.get('id')
                        if chunk_id and chunk_id != orig_chunk_id:
                             added_chunk_ids.add(chunk_id)
                else:
                    # If only 1 chunk found in DB, just add the original record back
                    processed_results.append(record) # Keep original single chunk
                    log_row[4] = "Keep Single (1 in DB)"
                    log_row[5] = 0
            except Exception as e:
                print(f"ERROR: Failed to fetch or process expansion for section {section_key}: {e}", file=sys.stderr)
                traceback.print_exc()
                processed_results.append(record) # Add original back on error
                log_row[4] = "Error - Keep Single"
                log_row[5] = 0
        else:
            # If no expansion needed, just add the original record
            processed_results.append(record)
            # log_row already defaults to "Keep Single", 0

        expansion_log_data.append(log_row)

    # Print the expansion log table
    try:
        print(tabulate(expansion_log_data, headers=headers_expansion, tablefmt="grid"))
    except ImportError:
        print("WARN: 'tabulate' library not found. Skipping table format.")
        for row in expansion_log_data:
            print(f"ID: {row[0]}, Rank: {row[1]}, Pages: {row[2]}, Threshold: {row[3]}, Action: {row[4]}, Added: {row[5]}")

    print(f"Finished section expansion. Result count: {len(processed_results)} (items/groups). Added {len(added_chunk_ids)} new chunks.")
    return processed_results, added_chunk_ids
def fill_page_gaps(cursor, results: list[dict | list[dict]], max_gap: int) -> tuple[list[dict | list[dict]], set]:
    """
    Identifies small page gaps between consecutive results and fetches missing chunks.
    Handles both single chunks and groups. Returns the updated list and a set of added chunk IDs.
    """
    print(f"\n--- Step 3: Filling page gaps (Max Gap: {max_gap} pages) ---")
    if len(results) < 2:
        return results, set() # Need at least two items to have a gap

    items_with_ranges = []
    gap_log_data = []
    headers_gaps = ["Between Item (Seq)", "And Item (Seq)", "Page Gap", "Action", "Added Chunks"]
    added_chunk_ids = set() # Track IDs added by this step
    for item in results:
        if isinstance(item, dict) and item.get('type') == 'group':
            # It's a group
            first_chunk = item['chunks'][0]
            last_chunk = item['chunks'][-1]
            doc_id = first_chunk.get('document_id')
            page_start = first_chunk.get('page_start')
            page_end = last_chunk.get('page_end')
            min_seq = first_chunk.get('sequence_number')
            max_seq = last_chunk.get('sequence_number')
            if all(v is not None for v in [doc_id, page_start, page_end, min_seq, max_seq]): # Corrected indentation
                items_with_ranges.append({
                    'item': item, 'doc_id': doc_id, 'page_start': page_start, 'page_end': page_end,
                    'min_seq': min_seq, 'max_seq': max_seq, 'is_group': True,
                            'id_repr': f"Group({min_seq}-{max_seq})" # Representation for logging
                        })
        elif isinstance(item, dict):
             # It's a single chunk
            doc_id = item.get('document_id')
            page_start = item.get('page_start')
            page_end = item.get('page_end')
            seq = item.get('sequence_number')
            chunk_id = item.get('id')
            if all(v is not None for v in [doc_id, page_start, page_end, seq]):
                 items_with_ranges.append({
                    'item': item, 'doc_id': doc_id, 'page_start': page_start, 'page_end': page_end,
                    'min_seq': seq, 'max_seq': seq, 'is_group': False,
                    'id_repr': f"Chunk({chunk_id})" # Representation for logging
                 })

    if len(items_with_ranges) < 2:
        print("  Not enough items with page ranges to check for gaps.")
        return results, set() # Not enough valid items with page info

    # Sort items by page_start primarily, then sequence number
    items_with_ranges.sort(key=lambda x: (x['page_start'], x['min_seq']))

    final_results_with_gaps = []
    last_item_info = None

    for current_item_info in items_with_ranges:
        if last_item_info:
            # Check for gap only if documents match
            if last_item_info['doc_id'] == current_item_info['doc_id']:
                page_gap = current_item_info['page_start'] - last_item_info['page_end'] - 1
                seq_gap = current_item_info['min_seq'] - last_item_info['max_seq'] - 1
                log_row = [f"{last_item_info['id_repr']} ({last_item_info['max_seq']})", f"{current_item_info['id_repr']} ({current_item_info['min_seq']})", page_gap, "None", 0]

                # Check if gap is within threshold and sequence numbers make sense
                if 0 < page_gap <= max_gap and seq_gap >= 0:
                    # print(f"  - Found gap of {page_gap} pages (Seq {last_item_info['max_seq']} to {current_item_info['min_seq']}). Fetching gap chunks.") # Replaced by table
                    try:
                        sql = """
                            SELECT * FROM textbook_chunks
                            WHERE document_id = %s AND sequence_number > %s AND sequence_number < %s
                            ORDER BY sequence_number;
                        """
                        cursor.execute(sql, (current_item_info['doc_id'], last_item_info['max_seq'], current_item_info['min_seq']))
                        gap_chunks_raw = cursor.fetchall()
                        num_added = len(gap_chunks_raw)
                        if num_added > 0:
                            gap_chunks = [dict(chunk) for chunk in gap_chunks_raw]
                            final_results_with_gaps.extend(gap_chunks) # Add gap chunks
                            log_row[3] = f"Fill Gap ({num_added} chunks)"
                            log_row[4] = num_added
                            # Track added IDs
                            for chunk in gap_chunks:
                                if chunk.get('id'): added_chunk_ids.add(chunk.get('id'))
                        else:
                            # print(f"    No chunks found in DB for gap between seq {last_item_info['max_seq']} and {current_item_info['min_seq']}.") # Replaced by table
                            log_row[3] = "No Chunks Found"
                            log_row[4] = 0
                    except Exception as e:
                        print(f"ERROR: Failed to fetch or process gap fill between seq {last_item_info['max_seq']} and {current_item_info['min_seq']}: {e}", file=sys.stderr)
                        traceback.print_exc()
                        log_row[3] = "Error Fetching"
                        log_row[4] = 0
                elif page_gap > max_gap:
                     log_row[3] = f"Gap > {max_gap} pages"
                     log_row[4] = 0
                else: # page_gap <= 0 or seq_gap < 0
                     log_row[3] = "No Gap / Overlap"
                     log_row[4] = 0
                gap_log_data.append(log_row)


        # Add the current item (chunk or group)
        final_results_with_gaps.append(current_item_info['item'])
        last_item_info = current_item_info

    # Print the gap log table
    if gap_log_data:
        try:
            print(tabulate(gap_log_data, headers=headers_gaps, tablefmt="grid"))
        except ImportError:
            print("WARN: 'tabulate' library not found. Skipping table format.")
            for row in gap_log_data:
                print(f"Between: {row[0]}, And: {row[1]}, Gap: {row[2]}, Action: {row[3]}, Added: {row[4]}")
    else:
        print("  No gaps checked (less than 2 items with page ranges).")


    print(f"Finished gap filling. Result count: {len(final_results_with_gaps)}. Added {len(added_chunk_ids)} new chunks.")
    return final_results_with_gaps, added_chunk_ids


def rerank_by_importance(results: list[dict | list[dict]], importance_factor: float) -> list[dict | list[dict]]:
    """
    Calculates a new score based on original combined score and importance score.
    Handles both single chunks and groups.
    """
    print(f"\n--- Step 4: Reranking by importance (Factor: {importance_factor}) ---")
    if not results: return []

    reranked_results = []
    for item in results:
        new_score = 0.0
        original_score = 0.0
        importance = 0.0

        if isinstance(item, dict) and item.get('type') == 'group':
            # Group: Use scores from the original triggering chunk
            original_score = item.get('original_combined_score', 0.0) or 0.0
            importance = item.get('original_importance_score', 0.0) or 0.0
            item_id = f"Group starting with chunk {item['chunks'][0].get('id', 'N/A')}" # For logging
        elif isinstance(item, dict):
            # Single chunk
            original_score = item.get('combined_score', 0.0) or 0.0
            importance = item.get('importance_score', 0.0) or 0.0
            item_id = f"Chunk {item.get('id', 'N/A')}" # For logging
        else:
             # Handle unexpected items (e.g., gap-filled chunks might not be dicts if fetch failed)
             if isinstance(item, psycopg2.extras.DictRow):
                 item = dict(item) # Convert if necessary
                 original_score = item.get('combined_score', 0.0) or 0.0 # Might be missing if it's a gap chunk
                 importance = item.get('importance_score', 0.0) or 0.0
                 item_id = f"Gap Chunk {item.get('id', 'N/A')}"
             else:
                print(f"WARNING: Skipping unexpected item type in reranking: {type(item)}", file=sys.stderr)
                continue

        # Calculate new score: new_score = original_score * (1 + factor * importance)
        # Ensure scores are floats
        try:
            original_score = float(original_score)
            importance = float(importance)
            boost = 1.0 + (importance_factor * importance)
            new_score = original_score * boost
            # print(f"  - {item_id}: Orig Score={original_score:.4f}, Importance={importance:.2f}, Boost={boost:.4f}, New Score={new_score:.4f}")
        except (TypeError, ValueError) as e:
            print(f"WARNING: Could not calculate score for {item_id} due to invalid numeric values (Orig: {original_score}, Importance: {importance}). Setting new_score to 0. Error: {e}", file=sys.stderr)
            new_score = 0.0 # Default score on error

        # Add the new score to the item (or group dict)
        if isinstance(item, dict):
            item['new_score'] = new_score
            reranked_results.append(item)
        # We don't need to handle DictRow separately here as it should have been converted

    print(f"Finished importance reranking for {len(reranked_results)} items.")
    return reranked_results


# --- Response Generation Functions ---

def format_chunks_as_cards(results: list[dict | list[dict]]):
    """
    Formats database results into "cards" for better GPT context understanding.
    
    Args:
        results: List of database result rows (as dict cursors)
    
    Returns:
        Formatted string with all chunks as cards
    """
    """
    Formats database results (including groups) into "cards" for GPT context.
    Includes only specified fields with labels.
    """
    print("\n--- Formatting Final Results as Cards for LLM ---")
    cards = []
    final_item_count = 0

    for i, item in enumerate(results):
        card_parts = []
        content_parts = []
        record_for_metadata = None
        is_group = False

        if isinstance(item, dict) and item.get('type') == 'group':
            # It's an expanded section group
            is_group = True
            if not item.get('chunks'): continue # Skip empty groups
            record_for_metadata = item['chunks'][0] # Use first chunk for metadata
            card_parts.append(f"--- CARD {i+1} (Reconstructed Section) ---")
            # Concatenate content from all chunks in the group
            for chunk in item['chunks']:
                 content_parts.append(chunk.get('content', ''))
            content = "\n\n".join(filter(None, content_parts))
            print(f"  - Formatting Card {i+1}: Group of {len(item['chunks'])} chunks (Section: {record_for_metadata.get('section_hierarchy', 'N/A')})")

        elif isinstance(item, dict):
             # It's a single chunk (original, or filled gap)
             # Handle potential DictRow from gap filling if not converted earlier
             if isinstance(item, psycopg2.extras.DictRow):
                 item = dict(item)
             record_for_metadata = item
             card_parts.append(f"--- CARD {i+1} ---")
             content = record_for_metadata.get('content', '')
             print(f"  - Formatting Card {i+1}: Single Chunk ID {record_for_metadata.get('id', 'N/A')}")
        else:
            print(f"WARNING: Skipping unexpected item type during formatting: {type(item)}", file=sys.stderr)
            continue

        if not record_for_metadata or not content:
            print(f"WARNING: Skipping Card {i+1} due to missing metadata or content.", file=sys.stderr)
            continue

        # Extract and format required fields with labels
        chapter_name = record_for_metadata.get('chapter_name', 'Unknown Chapter')
        section_title = record_for_metadata.get('section_title', 'Unknown Section')
        section_hierarchy = record_for_metadata.get('section_hierarchy', '')
        standard = record_for_metadata.get('standard')
        standard_codes = record_for_metadata.get('standard_codes')

        card_parts.append(f"Chapter: {chapter_name}")
        card_parts.append(f"Section Title: {section_title}")
        if section_hierarchy:
            card_parts.append(f"Section Hierarchy: {section_hierarchy}")
        if standard:
            card_parts.append(f"Standard: {standard}")
        if standard_codes and isinstance(standard_codes, list) and standard_codes:
            card_parts.append(f"Standard Codes: {', '.join(standard_codes)}")

        # Add the content
        card_parts.append("\nSection Content:")
        card_parts.append(content)

        cards.append("\n".join(card_parts))
        final_item_count += 1

    print(f"Formatted {final_item_count} cards.")
    # Join all cards with clear separation
    return "\n\n" + "\n\n".join(cards) + "\n\n"


def generate_response_from_chunks(client: OpenAI, query: str, formatted_chunks: str):
    """
    Generates a GPT response based on the query and formatted chunks.
    
    Args:
        client: OpenAI client
        query: User's query/question
        formatted_chunks: Chunks formatted as cards
    
    Returns:
        Generated response from GPT
    """
    """
    Generates a GPT response based on the query and formatted chunks.
    """
    print("\n--- Step 6: Generating Final Response from Processed Chunks ---")

    system_message = """You are a specialized accounting research assistant with expertise in IFRS and US GAAP standards.
Your task is to answer accounting questions based ONLY on the information provided in the context cards below.
Each card represents a relevant piece of text from the source document. Some cards might represent a reconstructed section containing multiple original text chunks.

Context Card Fields:
- Chapter: The name of the chapter the text belongs to.
- Section Title: The title of the specific section.
- Section Hierarchy: The structural path to the section (e.g., "Chapter 1 > Part A > Section 1.1").
- Standard: The primary accounting standard discussed (e.g., IFRS 16, ASC 842).
- Standard Codes: Specific codes or paragraph references within the standard.
- Section Content: The actual text content from the source document.

Instructions for Answering:
1. Rely EXCLUSIVELY on the "Section Content" provided in the cards. DO NOT use your external knowledge or training data.
2. Synthesize the information from the relevant cards to provide a comprehensive answer to the user's question.
3. You MUST cite your sources for every significant point or piece of information. Use the "Chapter" and "Section Title" or "Section Hierarchy" from the card(s) you used. Format citations clearly, e.g., [Source: Chapter Name, Section Title] or [Source: Section Hierarchy].
4. If multiple cards support a point, cite all relevant sources.
5. If the provided cards do not contain sufficient information to fully answer the question, clearly state what information is missing or cannot be determined from the context. Do not speculate or fabricate.
6. Structure your response logically, using headings or bullet points if helpful.
7. Provide a concise summary (2-3 sentences) at the end.

Remember: Accuracy and strict adherence to the provided context with proper citations are paramount."""

    user_message = f"""User Question: {query}

Context Cards:
{formatted_chunks}
---
Based ONLY on the context cards provided above, please answer the user's question with clear citations for each point, referencing the Chapter, Section Title, or Section Hierarchy."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            print(f"Making API call to {RESPONSE_MODEL} for final response generation (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...")

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

        except (APIError, RateLimitError, APITimeoutError) as e:
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


# --- Main Function ---

def generate_response(query: str):
    """
    Main function orchestrating the enhanced retrieval and response generation.
    """
    """
    Main function orchestrating the enhanced retrieval and response generation.
    """
    print("\n" + "="*80)
    print("--- Starting Enhanced Response Generation Process ---")
    print("="*80)
    print(f"\n>>> User Query:\n{query}\n")
    print(f"--- Configuration ---")
    print(f"Initial K Results: {INITIAL_K}")
    print(f"Final K Results: {'All' if FINAL_K is None else FINAL_K}")
    print(f"Importance Factor: {IMPORTANCE_FACTOR}")
    print(f"Relevance Model: {RELEVANCE_MODEL}")
    print(f"Response Model: {RESPONSE_MODEL}")
    print(f"Expansion Thresholds: General={SECTION_EXPANSION_GENERAL_THRESHOLD}pg, Expanded={SECTION_EXPANSION_EXPANDED_THRESHOLD}pg (Top {SECTION_EXPANSION_TOP_K_THRESHOLD} / Multi-Chunk)")
    print(f"Gap Fill Max Pages: {GAP_FILL_MAX_GAP}")
    print("-" * 80)


    # 1. Create OpenAI Client
    client = create_openai_client()
    if not client:
        return "ERROR: Failed to create OpenAI client."

    # 2. Generate Query Embedding
    query_embedding = generate_query_embedding(client, query, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    if query_embedding is None:
        return "ERROR: Failed to generate query embedding."

    # 3. Connect to Database
    conn = connect_to_db(DB_PARAMS)
    if conn is None:
        return "ERROR: Database connection failed."

    response_text = "Error: Processing failed before response generation."
    cursor = None
    processed_results = [] # To hold results through the pipeline

    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # --- Stage 1: Initial Retrieval ---
        print(f"\n--- Stage 1: Performing Initial Hybrid Search (Top {INITIAL_K}) ---")
        initial_results_raw = perform_hybrid_search(
            cursor=cursor,
            query=query,
            query_embedding=query_embedding,
            initial_k=INITIAL_K, # Use initial_k here
            doc_id=DOCUMENT_ID_TO_SEARCH
        )

        if not initial_results_raw:
            return "No relevant information found for your query in the initial database search."

        # Convert to list of dicts and add initial rank
        initial_results = []
        initial_chunk_ids = set() # Track initial IDs
        similarity_table_data = []
        headers = ["Rank", "Chunk ID", "Chapter", "Combined Score"]

        for i, row in enumerate(initial_results_raw):
            record = dict(row)
            chunk_id = record.get('id')
            rank = i + 1
            record['rank'] = rank
            initial_results.append(record)
            if chunk_id:
                initial_chunk_ids.add(chunk_id)
                similarity_table_data.append([
                    rank,
                    chunk_id,
                    record.get('chapter_name', 'N/A'),
                    f"{record.get('combined_score', 0.0):.4f}"
                ])

        print(f"Retrieved {len(initial_results)} initial results.")
        print("\n--- Initial Similarity Search Results ---")
        try:
            print(tabulate(similarity_table_data, headers=headers, tablefmt="grid"))
        except ImportError:
            print("WARN: 'tabulate' library not found. Install with 'pip install tabulate' for table formatting.")
            for row in similarity_table_data:
                print(f"Rank: {row[0]}, ID: {row[1]}, Chapter: {row[2]}, Score: {row[3]}")
        print("-" * 80)

        processed_results = initial_results
        all_added_chunk_ids = set() # Track IDs added by expansion/gaps
        all_removed_chunk_ids = set() # Track IDs removed by filtering

        # --- Stage 2: Summary Relevance Filtering ---
        # Modify filter_by_summary_relevance to return the map
        filtered_results, relevance_map = filter_by_summary_relevance(client, query, processed_results)
        if not filtered_results: return "No relevant information remained after summary filtering."

        # Log relevance filtering results
        relevance_table_data = []
        headers_relevance = ["Chunk ID", "Chapter", "Relevance (1=Yes, 0=No)", "Summary"]
        current_filtered_ids = set()
        for record in filtered_results:
             chunk_id = record.get('id')
             if chunk_id: current_filtered_ids.add(chunk_id)

        for record in initial_results: # Iterate original results to show all
             chunk_id = record.get('id')
             relevance_score = relevance_map.get(str(chunk_id), "N/A (Not Processed)")
             if relevance_score == 0:
                 all_removed_chunk_ids.add(chunk_id) # Track removed IDs
             relevance_table_data.append([
                 chunk_id,
                 record.get('chapter_name', 'N/A'),
                 relevance_score,
                 record.get('summary', '')[:100] + "..." # Truncate summary for display
             ])

        print("\n--- Summary Relevance Filtering Results ---")
        try:
            print(tabulate(relevance_table_data, headers=headers_relevance, tablefmt="grid"))
        except ImportError:
             print("WARN: 'tabulate' library not found. Skipping table format.")
             for row in relevance_table_data:
                 print(f"ID: {row[0]}, Chapter: {row[1]}, Relevant: {row[2]}, Summary: {row[3]}")
        print(f"Removed {len(all_removed_chunk_ids)} chunks based on summary relevance.")
        print("-" * 80)
        processed_results = filtered_results # Update results

        # --- Stage 3: Section Expansion ---
        # Modify expand_short_sections to return added IDs
        expanded_results, added_by_expansion_ids = expand_short_sections(
            cursor,
            processed_results, # Pass filtered results
            SECTION_EXPANSION_TOP_K_THRESHOLD,
            SECTION_EXPANSION_GENERAL_THRESHOLD,
            SECTION_EXPANSION_EXPANDED_THRESHOLD
        )
        if not expanded_results: return "No results remained after section expansion."
        all_added_chunk_ids.update(added_by_expansion_ids)
        print("-" * 80)
        processed_results = expanded_results # Update results

        # --- Stage 4: Gap Filling ---
        # Modify fill_page_gaps to return added IDs
        filled_results, added_by_gaps_ids = fill_page_gaps(cursor, processed_results, GAP_FILL_MAX_GAP)
        if not filled_results: return "No results remained after gap filling."
        all_added_chunk_ids.update(added_by_gaps_ids)
        print("-" * 80)
        processed_results = filled_results # Update results

        # --- Stage 5: Importance Reranking ---
        processed_results = rerank_by_importance(processed_results, IMPORTANCE_FACTOR)
        print("-" * 80)

        # --- Stage 6: Final Sorting ---
        print("\n--- Step 5: Sorting final results by new_score ---")
        processed_results.sort(key=lambda x: x.get('new_score', 0.0) if isinstance(x, dict) else 0.0, reverse=True)
        print(f"Sorted {len(processed_results)} final items.")
        print("-" * 80)

        # --- Stage 7: Truncation (Optional) ---
        if FINAL_K is not None and len(processed_results) > FINAL_K:
            print(f"\n--- Truncating results from {len(processed_results)} to {FINAL_K} ---")
            processed_results = processed_results[:FINAL_K]
            print("-" * 80)

        # --- Stage 7.5: Final Results Summary Log ---
        print("\n--- Final Results Summary ---")
        final_chunk_ids = set()
        for item in processed_results:
            if isinstance(item, dict) and item.get('type') == 'group':
                for chunk in item.get('chunks', []):
                    if chunk.get('id'): final_chunk_ids.add(chunk.get('id'))
            elif isinstance(item, dict) and item.get('id'):
                 final_chunk_ids.add(item.get('id'))
            elif isinstance(item, psycopg2.extras.DictRow): # Handle gap-filled chunks if not dicts
                 chunk_dict = dict(item)
                 if chunk_dict.get('id'): final_chunk_ids.add(chunk_dict.get('id'))


        print(f"Initial Chunk IDs ({len(initial_chunk_ids)}): {sorted(list(initial_chunk_ids))}")
        print(f"Removed by Filtering ({len(all_removed_chunk_ids)}): {sorted(list(all_removed_chunk_ids))}")
        # Added IDs are those in the final set that weren't in the initial set AND weren't removed
        truly_added_ids = final_chunk_ids - (initial_chunk_ids - all_removed_chunk_ids)
        print(f"Added by Expansion/Gaps ({len(truly_added_ids)}): {sorted(list(truly_added_ids))}")
        # Verify all_added_chunk_ids matches truly_added_ids (debugging check)
        # print(f"DEBUG: Tracked Added IDs ({len(all_added_chunk_ids)}): {sorted(list(all_added_chunk_ids))}")
        print(f"Final Chunk IDs in Context ({len(final_chunk_ids)}): {sorted(list(final_chunk_ids))}")
        print("-" * 80)


        # --- Stage 8: Format Cards ---
        formatted_chunks = format_chunks_as_cards(processed_results)
        print("-" * 80)

        # --- Stage 9: Generate Final Response ---
        response_text = generate_response_from_chunks(client, query, formatted_chunks)

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"ERROR: Enhanced response generation process failed: {error}", file=sys.stderr)
        traceback.print_exc()
        response_text = f"Error during response generation process: {str(error)}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n--- Enhanced Response Generation Process Finished ---")
    return response_text

# Example of how to use in a Jupyter notebook cell:
# response = generate_response("Explain the accounting treatment for leases under IFRS 16")
# print(response)
