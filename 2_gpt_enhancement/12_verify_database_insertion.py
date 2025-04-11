#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3F: Verify Database Insertion

Goal: Perform basic checks on the 'textbook_chunks' table to verify
the data inserted by the previous script (11_populate_textbook_chunks.py).

Checks:
- Total record count for the specific document_id.
- Count of records with non-NULL embeddings.
- Fetches and displays the first record for manual inspection.
- Checks maximum length of potentially truncated fields.
"""

import os
import sys
import json
import traceback
from pathlib import Path
import psycopg2
import psycopg2.extras # For DictCursor
from pgvector.psycopg2 import register_vector

# --- Configuration ---
# Source document ID to check
DOCUMENT_ID_TO_CHECK = "ey_international_gaap_2024"
# Optional: Path to the input JSON of script 11 to compare counts
# Set to None if you don't want to compare with the file count
INPUT_JSON_PATH_FOR_COUNT = Path("3D_final_records") / "final_database_records_with_embeddings.json"


# --- Database Configuration (Self-contained) ---
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "iris_dev",
    "password": "",  # No password needed for local development
}

print("--- Starting Database Insertion Verification ---")

# --- Helper Functions ---

def connect_to_db(params):
    """Connects to the PostgreSQL database."""
    conn = None
    try:
        print(f"Connecting to database '{params['dbname']}' on {params['host']}...")
        conn = psycopg2.connect(**params)
        # No autocommit needed for read-only operations
        print("Connection successful.")
        return conn
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

def get_input_record_count(filepath):
    """Loads records from the JSON input file and returns count."""
    if not filepath or not filepath.is_file():
        print(f"INFO: Input JSON file for count comparison not found or not specified: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            records = json.load(f)
        # Consider filtering out records that would have been skipped by script 11?
        # For now, just return total count for basic comparison.
        return len(records)
    except Exception as e:
        print(f"ERROR: Failed to load or count records from {filepath}: {e}", file=sys.stderr)
        return None

# --- Verification Functions ---

def check_record_count(cursor, doc_id):
    """Counts total records for the given document_id."""
    print(f"\n--- Checking Record Count for document_id='{doc_id}' ---")
    try:
        cursor.execute("SELECT COUNT(*) FROM textbook_chunks WHERE document_id = %s;", (doc_id,))
        count = cursor.fetchone()[0]
        print(f"Total records found in DB: {count}")
        return count
    except Exception as e:
        print(f"ERROR: Failed to count records: {e}", file=sys.stderr)
        return None

def check_embedding_count(cursor, doc_id):
    """Counts records with non-NULL embeddings for the given document_id."""
    print(f"\n--- Checking Embedding Count for document_id='{doc_id}' ---")
    try:
        cursor.execute("SELECT COUNT(*) FROM textbook_chunks WHERE document_id = %s AND embedding IS NOT NULL;", (doc_id,))
        count = cursor.fetchone()[0]
        print(f"Records with non-NULL embeddings: {count}")
        return count
    except Exception as e:
        print(f"ERROR: Failed to count embeddings: {e}", file=sys.stderr)
        return None

def fetch_first_record(cursor, doc_id):
    """Fetches the first record based on sequence_number for inspection."""
    print(f"\n--- Fetching First Record (by sequence_number) for document_id='{doc_id}' ---")
    try:
        # Use DictCursor to access columns by name
        cursor.execute("""
            SELECT * FROM textbook_chunks
            WHERE document_id = %s
            ORDER BY sequence_number ASC
            LIMIT 1;
        """, (doc_id,))
        record = cursor.fetchone()
        if record:
            print("First record data:")
            # Print selected fields for clarity
            print(f"  ID: {record.get('id')}")
            print(f"  Sequence Number: {record.get('sequence_number')}")
            print(f"  Document ID: {record.get('document_id')}")
            print(f"  Chapter Name: {record.get('chapter_name')}")
            print(f"  Tags: {record.get('tags')}")
            print(f"  Standard: {record.get('standard')}")
            print(f"  Standard Codes: {record.get('standard_codes')}")
            print(f"  Embedding Status: {'Present' if record.get('embedding') is not None else 'NULL'}")
            # print(f"  Embedding Preview: {str(record.get('embedding'))[:100]}...") # Optional: Show embedding preview
            print(f"  Content Preview: {str(record.get('content'))[:150]}...")
            print(f"  Section Hierarchy: {record.get('section_hierarchy')}")
            print(f"  Section Title: {record.get('section_title')}")
        else:
            print("No records found.")
        return record
    except Exception as e:
        print(f"ERROR: Failed to fetch first record: {e}", file=sys.stderr)
        return None

def check_max_lengths(cursor, doc_id):
    """Checks the maximum length of potentially truncated fields."""
    print(f"\n--- Checking Max Lengths for document_id='{doc_id}' ---")
    try:
        cursor.execute("""
            SELECT MAX(LENGTH(section_hierarchy)), MAX(LENGTH(section_title))
            FROM textbook_chunks
            WHERE document_id = %s;
        """, (doc_id,))
        max_hierarchy, max_title = cursor.fetchone()
        print(f"Max length of 'section_hierarchy': {max_hierarchy}")
        print(f"Max length of 'section_title': {max_title}")
        if max_hierarchy == 500:
            print("  INFO: Max length for section_hierarchy reached limit (500), truncation may have occurred.")
        if max_title == 500:
            print("  INFO: Max length for section_title reached limit (500), truncation may have occurred.")
        return max_hierarchy, max_title
    except Exception as e:
        print(f"ERROR: Failed to check max lengths: {e}", file=sys.stderr)
        return None, None

def check_text_search_setup(cursor, doc_id):
    """Verifies the text_search_vector column setup."""
    print(f"\n--- Checking Text Search Setup for document_id='{doc_id}' ---")
    try:
        # Check if column exists and if it's a generated column
        cursor.execute("""
            SELECT c.column_name, c.data_type, c.is_generated, c.generation_expression
            FROM information_schema.columns c
            WHERE c.table_name = 'textbook_chunks' AND c.column_name = 'text_search_vector';
        """)
        column_info = cursor.fetchone()
        
        if not column_info:
            print("ERROR: text_search_vector column not found in the table!")
            return False
        
        if column_info[2] == 'ALWAYS':
            print(f"✅ text_search_vector is a GENERATED ALWAYS column (type: {column_info[1]})")
            print(f"   Generation expression: {column_info[3]}")
        else:
            print(f"✅ text_search_vector is a regular column (type: {column_info[1]})")
        
        # Check if vectors are populated for this document
        cursor.execute("""
            SELECT COUNT(*), COUNT(text_search_vector)
            FROM textbook_chunks
            WHERE document_id = %s;
        """, (doc_id,))
        total_count, vector_count = cursor.fetchone()
        print(f"Total records: {total_count}, Records with text vectors: {vector_count}")
        
        if vector_count < total_count:
            print(f"WARNING: {total_count - vector_count} records have NULL text_search_vector")
        else:
            print("✅ All records have text search vectors populated")
        
        # Check for index
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'textbook_chunks' AND indexdef LIKE '%text_search_vector%';
        """)
        index = cursor.fetchone()
        
        if not index:
            print("WARNING: No index found on text_search_vector column!")
        else:
            print(f"✅ Text search index exists: {index[0]}")
        
        # Run a sample text search query to verify functionality
        cursor.execute("""
            SELECT COUNT(*) FROM textbook_chunks 
            WHERE document_id = %s AND text_search_vector @@ to_tsquery('english', 'accounting');
        """, (doc_id,))
        search_count = cursor.fetchone()[0]
        print(f"Sample keyword search for 'accounting' found {search_count} matches")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to check text search setup: {e}", file=sys.stderr)
        return False

# --- Main Execution Function ---
def verify_insertion():
    """Connects to DB and runs verification checks."""
    conn = connect_to_db(DB_PARAMS)
    if conn is None:
        print("Aborting due to database connection failure.", file=sys.stderr)
        raise RuntimeError("Database connection failed.")

    # Use DictCursor for easier column access by name
    cursor = None
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # Run checks
        db_count = check_record_count(cursor, DOCUMENT_ID_TO_CHECK)
        embedding_count = check_embedding_count(cursor, DOCUMENT_ID_TO_CHECK)
        fetch_first_record(cursor, DOCUMENT_ID_TO_CHECK)
        check_max_lengths(cursor, DOCUMENT_ID_TO_CHECK)
        check_text_search_setup(cursor, DOCUMENT_ID_TO_CHECK)

        # Compare DB count with input file count (optional)
        if INPUT_JSON_PATH_FOR_COUNT:
            input_count = get_input_record_count(INPUT_JSON_PATH_FOR_COUNT)
            if input_count is not None and db_count is not None:
                print(f"\n--- Count Comparison ---")
                print(f"Input JSON record count: {input_count}")
                print(f"Database record count:   {db_count}")
                if input_count == db_count:
                    print("Counts match.")
                else:
                    # Note: Counts might differ if script 11 skipped records during preparation
                    print("WARNING: Counts do NOT match. This might be expected if script 11 skipped invalid records during preparation.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"ERROR: Database verification failed: {error}", file=sys.stderr)
        traceback.print_exc()
        # No rollback needed for read operations
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n--- Database Insertion Verification Finished ---")


# --- To run in a notebook cell, call the function: ---
# verify_insertion()

# --- Or run as a script ---
if __name__ == "__main__":
    try:
        verify_insertion()
    except Exception as e:
        # Catch exceptions raised by verify_insertion for cleaner script exit
        print(f"FATAL ERROR during verification: {e}", file=sys.stderr)
        sys.exit(1)
