#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3D: Generate Embeddings (Notebook Version)

Goal: Generate embeddings for the assembled records using the specified model
and dimensions via the custom OpenAI endpoint.

Input:
- Assembled records JSON file from INPUT_FILE (e.g., 3C_pre_embedding_records/pre_embedding_records.json).

Output:
- A final JSON file containing the records with the 'embedding' field populated,
  saved to OUTPUT_FILE (e.g., 3D_final_records/final_database_records_with_embeddings.json).
"""

import os
import json
import time
import traceback
import requests # Added for _get_oauth_token
from pathlib import Path
from openai import OpenAI, APIError # Added OpenAI, APIError
from tqdm.notebook import tqdm # Use tqdm.notebook for Jupyter

# --- Configuration ---
# Adjust these paths if your notebook is not in the project root directory
INPUT_DIR = "3C_pre_embedding_records"
INPUT_FILENAME = "pre_embedding_records.json"
OUTPUT_DIR = "3D_final_records"
OUTPUT_FILENAME = "final_database_records_with_embeddings.json"

# --- Embedding Configuration ---
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 2000

# --- API Connection Configuration (Copied from Script 8) ---
BASE_URL = "https://api.example.com/v1" # Replace with actual endpoint if different

# OAuth settings - **LOAD SECURELY (e.g., environment variables)**
OAUTH_URL = "https://api.example.com/oauth/token" # Replace if needed
CLIENT_ID = os.environ.get("RBC_OAUTH_CLIENT_ID", "your_client_id") # Placeholder
CLIENT_SECRET = os.environ.get("RBC_OAUTH_CLIENT_SECRET", "your_client_secret") # Placeholder

# SSL certificate settings - **Ensure paths are correct**
SSL_SOURCE_PATH = os.environ.get("RBC_SSL_SOURCE_PATH", "/path/to/your/rbc-ca-bundle.cer") # Placeholder
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer" # Temporary local path

# --- Internal Constants (Copied/Adapted from Script 8) ---
_SSL_CONFIGURED = False # Flag to avoid redundant SSL setup
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 5 # seconds

print("--- Starting Embedding Generation (Notebook Version) ---")

# --- Helper Functions (Copied/Adapted from Script 8) ---

def _setup_ssl(source_path=SSL_SOURCE_PATH, local_path=SSL_LOCAL_PATH):
    """Copies SSL cert locally and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True # Already configured

    print("Setting up SSL certificate...") # Use print
    try:
        source = Path(source_path)
        local = Path(local_path)

        if not source.is_file():
            print(f"ERROR: SSL source certificate not found at {source_path}") # Use print
            return False

        local.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as source_file:
            content = source_file.read()
        with open(local, "wb") as dest_file:
            dest_file.write(content)

        os.environ["SSL_CERT_FILE"] = str(local)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local)
        print(f"SSL certificate configured successfully at: {local}") # Use print
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        print(f"ERROR: Error setting up SSL certificate: {e}") # Use print
        # traceback.print_exc() # Optional
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH):
    """Retrieves OAuth token from the specified endpoint."""
    print("Attempting to get OAuth token...") # Use print
    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    try:
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
            print("ERROR: 'access_token' not found in OAuth response.") # Use print
            return None
        print("OAuth token obtained successfully.") # Use print
        return oauth_token
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error getting OAuth token: {e}") # Use print
        # traceback.print_exc() # Optional
        return None

def create_openai_client(base_url=BASE_URL):
    """Sets up SSL, gets OAuth token, and creates the OpenAI client."""
    if not _setup_ssl():
        print("ERROR: Aborting client creation due to SSL setup failure.") # Use print
        return None # SSL setup failed

    api_key = _get_oauth_token()
    if not api_key:
        print("ERROR: Aborting client creation due to OAuth token failure.") # Use print
        return None # Token retrieval failed

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("OpenAI client created successfully.") # Use print
        return client
    except Exception as e:
        print(f"ERROR: Error creating OpenAI client: {e}") # Use print
        # traceback.print_exc() # Optional
        return None

def _call_embedding_api_with_retry(client, text_inputs: list[str], model: str, dimensions: int) -> list | None:
    """Calls the OpenAI embedding API with a batch of texts and retry logic."""
    last_exception = None
    if not text_inputs:
        print("WARNING: Empty batch received for embedding. Returning None.")
        return None
    # Ensure all inputs are strings and non-empty, replace empty strings if necessary
    processed_inputs = [text if text else " " for text in text_inputs]


    for attempt in range(_RETRY_ATTEMPTS):
        try:
            # print(f"DEBUG: Calling embedding API for batch size {len(processed_inputs)} (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...") # Optional debug
            response = client.embeddings.create(
                input=processed_inputs, # Send the batch of texts
                model=model,
                dimensions=dimensions
            )
            # Extract the list of embedding vectors
            if response.data and len(response.data) == len(processed_inputs):
                # Return list of embeddings, preserving order
                return [item.embedding for item in response.data]
            else:
                # This case might happen if the API returns fewer embeddings than inputs, which is unexpected.
                print(f"ERROR: Embedding API returned {len(response.data)} embeddings for {len(processed_inputs)} inputs.")
                raise ValueError("Mismatch between input batch size and embedding response size.")

        except APIError as e:
            print(f"WARNING: API Error on embedding batch attempt {attempt + 1}: {e}") # Use print
            last_exception = e
            # Consider different delays for different errors (e.g., rate limits)
            delay = _RETRY_DELAY * (attempt + 1)
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"WARNING: Non-API Error on embedding batch attempt {attempt + 1}: {e}") # Use print
            last_exception = e
            # traceback.print_exc() # Optional
            delay = _RETRY_DELAY # Shorter delay for non-API errors
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    print(f"ERROR: Embedding API call failed after {_RETRY_ATTEMPTS} attempts for batch starting with: {processed_inputs[0][:100]}...") # Use print
    return None


# --- Main Embedding Generation Logic (Top Level for Notebook) ---

# 1. Define Paths
input_file = Path(INPUT_DIR) / INPUT_FILENAME
output_path = Path(OUTPUT_DIR)
output_file = output_path / OUTPUT_FILENAME

# 2. Load Input Data
records = []
if input_file.is_file():
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"Successfully loaded {len(records)} records from {input_file}") # Use print
    except json.JSONDecodeError:
        print(f"ERROR: Failed to decode JSON from {input_file}. Aborting.") # Use print
        records = None # Signal error
    except Exception as e:
        print(f"ERROR: Failed to load input file {input_file}: {e}") # Use print
        records = None # Signal error
else:
    print(f"ERROR: Input file not found: {input_file}. Aborting.") # Use print
    records = None # Signal error

# Proceed only if records were loaded successfully
if records is not None:
    # 3. Create OpenAI client
    client = create_openai_client(BASE_URL)

    if not client:
        print("ERROR: Failed to create OpenAI client. Aborting embedding generation.") # Use print
        records = None # Signal error

# Proceed only if client was created
if records is not None and client is not None:
    # 4. Create output directory
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_path}") # Use print
    except Exception as e:
        print(f"ERROR: Failed to create output directory {output_path}: {e}") # Use print
        records = None # Signal error

# Proceed only if output directory is ready
if records is not None and client is not None:
    # 5. Generate Embeddings
    print(f"Generating embeddings for {len(records)} records using model '{EMBEDDING_MODEL}' with {EMBEDDING_DIMENSIONS} dimensions...") # Use print
    embeddings_generated = 0
    embeddings_failed = 0
    BATCH_SIZE = 50 # Process records in batches of 50

    print(f"Generating embeddings for {len(records)} records using model '{EMBEDDING_MODEL}' with {EMBEDDING_DIMENSIONS} dimensions (Batch Size: {BATCH_SIZE})...") # Use print

    # Process in batches
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Generating Embeddings Batches"):
        batch_records = records[i:i + BATCH_SIZE]
        batch_texts = [record.get('content', '') for record in batch_records] # Get content, default to empty string

        # Check if all texts in batch are empty before calling API
        if not any(batch_texts):
             print(f"WARNING: Batch {i//BATCH_SIZE + 1} contains only empty content. Skipping API call.")
             for record in batch_records:
                 record['embedding'] = None
                 embeddings_failed += 1
             continue

        try:
            embedding_vectors = _call_embedding_api_with_retry(
                client=client,
                text_inputs=batch_texts,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS
            )

            if embedding_vectors and len(embedding_vectors) == len(batch_records):
                # Assign embeddings back to the records in the batch
                for record, vector in zip(batch_records, embedding_vectors):
                    record['embedding'] = vector
                embeddings_generated += len(batch_records)
            else:
                # Handle case where API call failed or returned unexpected results for the batch
                print(f"ERROR: Failed to get embeddings for batch starting at index {i}. Setting embeddings to None for this batch.")
                for record in batch_records:
                    record['embedding'] = None
                embeddings_failed += len(batch_records)

        except Exception as e:
            # Catch any unexpected errors during batch processing
            print(f"ERROR: Unexpected error processing batch starting at index {i}: {e}") # Use print
            # traceback.print_exc() # Optional
            for record in batch_records:
                record['embedding'] = None # Ensure field exists but is None on error
            embeddings_failed += len(batch_records)


    print(f"Finished generating embeddings.") # Use print
    print(f"  Total Records Processed: {len(records)}")
    print(f"  Embeddings Successfully Generated: {embeddings_generated}")
    print(f"  Failed/Skipped: {embeddings_failed}")

    # 6. Save Final Output
    if embeddings_generated > 0 or embeddings_failed == 0: # Save even if some failed, unless all failed
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False) # Use indent=2 for readability
            print(f"Successfully saved {len(records)} records (with embeddings) to {output_file}") # Use print
        except Exception as e:
            print(f"ERROR: Failed to save output file {output_file}: {e}") # Use print
    else:
        print("Skipping save as no embeddings were successfully generated.")

print("--- Embedding Generation Finished (Notebook Version) ---")
