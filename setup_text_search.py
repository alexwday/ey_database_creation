#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Set up PostgreSQL full-text search for the textbook_chunks table.
This script:
1. Adds a text_search_vector column if it doesn't exist
2. Updates it with tsvector data from the content column
3. Creates a GIN index for faster searching
"""

import psycopg2
import sys
import time

# Database configuration
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "iris_dev",
    "password": "",  # No password needed for local development
}

def setup_text_search():
    """Set up full-text search in PostgreSQL."""
    print("Connecting to PostgreSQL database...")
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = False  # Use transaction
        cursor = conn.cursor()
        
        # Step 1: Check if text_search_vector column exists
        print("Checking if text_search_vector column exists...")
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'textbook_chunks' AND column_name = 'text_search_vector';
        """)
        if not cursor.fetchone():
            print("Creating text_search_vector column...")
            cursor.execute("ALTER TABLE textbook_chunks ADD COLUMN text_search_vector tsvector;")
            print("Column created successfully.")
        else:
            print("text_search_vector column already exists.")

        # Step 2: Update text_search_vector with content data
        print("Updating text_search_vector column from content...")
        start_time = time.time()
        cursor.execute("""
            UPDATE textbook_chunks 
            SET text_search_vector = to_tsvector('english', 
                COALESCE(content, '') || ' ' || 
                COALESCE(section_title, '') || ' ' || 
                COALESCE(chapter_name, '')
            );
        """)
        rows_updated = cursor.rowcount
        duration = time.time() - start_time
        print(f"Updated {rows_updated} rows in {duration:.2f} seconds.")
        
        # Step 3: Check if index exists
        print("Checking if text search index exists...")
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'textbook_chunks' AND indexdef LIKE '%text_search_vector%';
        """)
        if not cursor.fetchone():
            print("Creating GIN index on text_search_vector (this may take a while)...")
            start_time = time.time()
            cursor.execute("CREATE INDEX idx_textbook_chunks_tsv ON textbook_chunks USING GIN(text_search_vector);")
            duration = time.time() - start_time
            print(f"GIN index created in {duration:.2f} seconds.")
        else:
            print("GIN index already exists for text_search_vector.")
            
        # Commit the transaction
        conn.commit()
        print("✅ Text search setup completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
            print("Changes rolled back due to error.")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("Database connection closed.")
    
    return True

if __name__ == "__main__":
    setup_text_search()