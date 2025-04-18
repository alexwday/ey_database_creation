#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3E: Populate Textbook Chunks Table

Goal: Load the final records (including embeddings) into the PostgreSQL
'textbook_chunks' table. Deletes existing records for the specific
document_id before inserting new ones.

Input:
- Final records JSON file from INPUT_FILE (e.g., 3D_final_records/final_database_records_with_embeddings.json).

Output:
- Populates the 'textbook_chunks' table in the database.
"""

import os
import json
import sys
import traceback
from pathlib import Path
import psycopg2
import psycopg2.extras # For execute_values

# --- Configuration ---
INPUT_DIR = "3D_final_records"
INPUT_FILENAME = "final_database_records_with_embeddings.json"
DOCUMENT_ID_TO_REPLACE = "ey_international_gaap_2024" # Document ID to delete before inserting

# --- Database Configuration (Self-contained) ---
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "iris_dev",
    "password": "",  # No password needed for local development
}

print("--- Starting Textbook Chunk Population ---")

# --- Helper Functions ---

def connect_to_db(params):
    """Connects to the PostgreSQL database."""
    conn = None
    try:
        print(f"Connecting to database '{params['dbname']}' on {params['host']}...")
        conn = psycopg2.connect(**params)
        conn.autocommit = False # Use transactions
        print("Connection successful.")
        return conn
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

def load_records(filepath):
    """Loads records from the JSON input file."""
    if not filepath.is_file():
        print(f"ERROR: Input file not found: {filepath}", file=sys.stderr)
        return None
    try:
        print(f"Loading records from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"Successfully loaded {len(records)} records.")
        return records
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON from {filepath}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Failed to load input file {filepath}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

def prepare_data_for_insert(records):
    """Prepares data tuples for batch insertion, matching table columns."""
    data_tuples = []
    print("Preparing data for insertion...")
    skipped_count = 0
    # Keys expected in the JSON record based on script 9/10 and corrected schema
    required_keys = [
        "document_id", "chapter_name", "tags", "standard", "standard_codes",
        "embedding", "sequence_number", "section_references", "page_start",
        "page_end", "summary", "importance_score", "section_hierarchy",
        "section_title",
        "content"
    ]

    for i, record in enumerate(records):
        # Validate required keys are present
        missing_keys = [key for key in required_keys if key not in record]
        if missing_keys:
            print(f"WARNING: Record {i} missing keys: {missing_keys}. Skipping.", file=sys.stderr)
            skipped_count += 1
            continue

        # Handle potential None values for numeric/float types if necessary
        page_start = record.get("page_start")
        page_end = record.get("page_end")
        importance_score = record.get("importance_score")
        sequence_number = record.get("sequence_number")

        # Ensure embedding is a list or None
        embedding = record.get("embedding")
        if embedding is not None and not isinstance(embedding, list):
             print(f"WARNING: Record {i} has invalid embedding type ({type(embedding)}). Setting to NULL. Skipping.", file=sys.stderr)
             skipped_count += 1
             continue # Skip if embedding is invalid

        # Ensure content is not None or empty
        content = record.get("content")
        if not content:
             print(f"WARNING: Record {i} has empty content. Skipping.", file=sys.stderr)
             skipped_count += 1
             continue

        # Prepare and truncate potentially long fields before appending
        section_hierarchy_orig = record.get("section_hierarchy") or ""
        section_title_orig = record.get("section_title") or ""
        section_hierarchy_trunc = section_hierarchy_orig[:500]
        section_title_trunc = section_title_orig[:500]

        # Order must match the INSERT statement columns below
        data_tuples.append((
            record.get("document_id"),
            record.get("chapter_name"),
            record.get("tags"), # TEXT[]
            record.get("standard"), # VARCHAR(100)
            record.get("standard_codes"), # TEXT[]
            embedding, # VECTOR(2000)
            sequence_number, # INT
            record.get("section_references"), # TEXT[]
            page_start, # INT
            page_end, # INT
            record.get("summary"), # TEXT
            importance_score, # FLOAT
            section_hierarchy_trunc, # Use truncated value
            section_title_trunc, # Use truncated value
            content # TEXT NOT NULL
        ))

        # Log truncation events (using original values for length check)
        if len(section_hierarchy_orig) > 500:
            print(f"WARNING: Record sequence_number={sequence_number}: Truncated 'section_hierarchy' (orig_len={len(section_hierarchy_orig)})", file=sys.stderr)
        if len(section_title_orig) > 500:
            print(f"WARNING: Record sequence_number={sequence_number}: Truncated 'section_title' (orig_len={len(section_title_orig)})", file=sys.stderr)


    if skipped_count > 0:
         print(f"WARNING: Skipped {skipped_count} records due to missing keys, invalid embedding, or empty content.", file=sys.stderr)
    print(f"Prepared {len(data_tuples)} records for insertion.")
    return data_tuples

def populate_textbook_chunks():
    """Loads records, connects to DB, deletes old, inserts new."""
    input_file_path = Path(INPUT_DIR) / INPUT_FILENAME
    records_to_insert = load_records(input_file_path)

    if records_to_insert is None:
        print("ERROR: Aborting due to failure loading records.", file=sys.stderr)
        raise RuntimeError("Failed to load records.")

    if not records_to_insert:
        print("No records found in the input file. Exiting.")
        return # Nothing to do

    conn = connect_to_db(DB_PARAMS)
    if conn is None:
        print("ERROR: Aborting due to database connection failure.", file=sys.stderr)
        raise RuntimeError("Database connection failed.")

    cursor = None # Initialize cursor to None
    try:
        cursor = conn.cursor()

        # 0. Ensure text_search_vector column exists (for keyword search)
        print("Ensuring text_search_vector column exists...")
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'textbook_chunks' AND column_name = 'text_search_vector';
        """)
        if not cursor.fetchone():
            print("Adding text_search_vector column to the table...")
            cursor.execute("ALTER TABLE textbook_chunks ADD COLUMN text_search_vector tsvector;")
            print("Column added successfully.")
        
        # 1. Delete existing records for the specified document_id
        print(f"Deleting existing records for document_id: '{DOCUMENT_ID_TO_REPLACE}'...")
        delete_sql = "DELETE FROM textbook_chunks WHERE document_id = %s;"
        cursor.execute(delete_sql, (DOCUMENT_ID_TO_REPLACE,))
        deleted_count = cursor.rowcount
        print(f"Deleted {deleted_count} existing records.")

        # 2. Prepare data for insertion
        data_tuples = prepare_data_for_insert(records_to_insert)

        if not data_tuples:
            print("ERROR: No valid records prepared for insertion. Aborting.", file=sys.stderr)
            conn.rollback() # Rollback deletion if no new records to insert
            raise ValueError("No valid records prepared for insertion.")

        # 3. Insert new records using execute_values for efficiency
        print(f"Inserting {len(data_tuples)} new records...")
        # IMPORTANT: Column order must match the data_tuples order
        insert_sql = """
            INSERT INTO textbook_chunks (
                document_id, chapter_name, tags, standard, standard_codes,
                embedding, sequence_number, section_references, page_start,
                page_end, summary, importance_score, section_hierarchy,
                section_title, content
            ) VALUES %s
            RETURNING id;
        """
        # Execute the insert and get the IDs
        inserted_ids = psycopg2.extras.execute_values(
            cursor,
            insert_sql,
            data_tuples,
            template=None,
            page_size=100, # Adjust batch size as needed
            fetch=True  # Return the inserted IDs
        )
        inserted_count = len(inserted_ids) if inserted_ids else cursor.rowcount
        print(f"Successfully inserted {inserted_count} records.")
        
        # 4. Check if text_search_vector is a generated column
        print("Checking if text_search_vector needs manual updating...")
        cursor.execute("""
            SELECT generation_expression 
            FROM information_schema.columns 
            WHERE table_name = 'textbook_chunks' 
            AND column_name = 'text_search_vector' 
            AND is_generated = 'ALWAYS';
        """)
        is_generated = cursor.fetchone()
        
        if is_generated:
            # If it's a generated column, it updates automatically - no need to do anything
            print(f"✅ text_search_vector is a generated column with expression: {is_generated[0]}")
            print("  No manual update needed - values are generated automatically")
        else:
            # If it's a regular column, we need to update it manually
            print("Updating text_search_vector for full-text search capabilities...")
            try:
                update_sql = """
                    UPDATE textbook_chunks 
                    SET text_search_vector = to_tsvector('english', 
                        COALESCE(content, '') || ' ' || 
                        COALESCE(section_title, '') || ' ' || 
                        COALESCE(chapter_name, '')
                    )
                    WHERE document_id = %s;
                """
                cursor.execute(update_sql, (DOCUMENT_ID_TO_REPLACE,))
                updated_count = cursor.rowcount
                print(f"Updated text_search_vector for {updated_count} records.")
            except Exception as e:
                # If update fails, warn but continue - it might be a generated column with different settings
                print(f"WARNING: Could not update text_search_vector manually: {e}")
                print("  This is expected if the column is managed by a trigger or is a generated column")
                print("  The database might be handling text search vectors automatically")
        
        # 5. Ensure text search index exists
        print("Checking if text search index exists...")
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'textbook_chunks' AND indexdef LIKE '%text_search_vector%';
        """)
        if not cursor.fetchone():
            print("Creating GIN index on text_search_vector...")
            cursor.execute("CREATE INDEX idx_textbook_chunks_tsv ON textbook_chunks USING GIN(text_search_vector);")
            print("GIN index created for faster text search.")

        # 6. Commit transaction
        print("Committing transaction...")
        conn.commit()
        print("Transaction committed successfully.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"ERROR: Database operation failed: {error}", file=sys.stderr)
        traceback.print_exc()
        if conn:
            print("Rolling back transaction...")
            conn.rollback()
        raise error # Re-raise the caught exception
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("Database connection closed.")

    print("--- Textbook Chunk Population Finished ---")

# --- To run in a notebook cell, call the function: ---
# populate_textbook_chunks()
