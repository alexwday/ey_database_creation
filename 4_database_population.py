#!/usr/bin/env python3
"""
Stage 4: Database population and verification

This script loads the final JSON chunks with embeddings, clears any existing data
from the database for the same document_id, inserts the new chunks, and verifies
the insertion.

Input: JSON chunk files with embeddings from Stage 3
Output: Populated database with verified records
"""

import os
import json
import traceback
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values
from tqdm import tqdm
import random
from pathlib import Path

# --- Configuration ---
INPUT_DIR = "database_chunks"  # Directory with final JSON chunks from Stage 3

# Database configuration
DB_NAME = os.environ.get("DB_NAME", "guidance_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

# --- Database Schema ---
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS guidance_sections (
  -- SYSTEM FIELDS
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- STRUCTURAL POSITIONING FIELDS
  document_id TEXT,
  chapter_number INT,
  section_number INT,
  part_number INT,
  sequence_number INT,

  -- CHAPTER-LEVEL METADATA
  chapter_name TEXT,
  chapter_tags TEXT[],
  chapter_summary TEXT,
  chapter_token_count INT,

  -- SECTION-LEVEL PAGINATION & IMPORTANCE
  section_start_page INT,
  section_end_page INT,
  section_importance_score FLOAT,
  section_token_count INT,

  -- SECTION-LEVEL METADATA
  section_hierarchy TEXT,
  section_title TEXT,
  section_standard TEXT,
  section_standard_codes TEXT[],
  section_references TEXT[],

  -- CONTENT & EMBEDDING
  content TEXT NOT NULL,
  embedding VECTOR(2000),
  text_search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_guidance_sections_document_id ON guidance_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_guidance_sections_sequence_number ON guidance_sections(sequence_number);
CREATE INDEX IF NOT EXISTS idx_guidance_sections_chapter_number ON guidance_sections(chapter_number);
CREATE INDEX IF NOT EXISTS idx_guidance_sections_text_search ON guidance_sections USING GIN(text_search_vector);

-- Create vector index if pgvector is installed
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace 
                      WHERE c.relname = 'idx_guidance_sections_embedding' AND n.nspname = 'public') THEN
            CREATE INDEX idx_guidance_sections_embedding ON guidance_sections USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        END IF;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Failed to create vector index: %', SQLERRM;
END$$;
"""

# --- Utility Functions ---

def create_db_connection():
    """Creates and returns a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("Database connection established successfully.")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        traceback.print_exc()
        return None

def setup_database(conn):
    """Sets up the database schema if it doesn't exist."""
    try:
        cursor = conn.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        has_pgvector = cursor.fetchone()[0]
        
        if not has_pgvector:
            print("Warning: pgvector extension not found. Attempting to install...")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                print("pgvector extension installed successfully.")
            except Exception as e:
                print(f"Error installing pgvector extension: {e}")
                print("Please install pgvector extension manually or vector operations will not work.")
        
        # Create schema
        cursor.execute(SCHEMA_SQL)
        conn.commit()
        print("Database schema setup complete.")
        return True
    except Exception as e:
        print(f"Error setting up database schema: {e}")
        traceback.print_exc()
        conn.rollback()
        return False

def clear_existing_data(conn, document_id):
    """Clears existing data for the given document_id."""
    try:
        cursor = conn.cursor()
        query = "DELETE FROM guidance_sections WHERE document_id = %s"
        cursor.execute(query, (document_id,))
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"Cleared {deleted_count} existing records for document_id: {document_id}")
        return True
    except Exception as e:
        print(f"Error clearing existing data: {e}")
        traceback.print_exc()
        conn.rollback()
        return False

def insert_chunks(conn, chunks):
    """Inserts chunks into the database."""
    if not chunks:
        print("No chunks to insert.")
        return 0
    
    print(f"Preparing to insert {len(chunks)} chunks...")
    inserted_count = 0
    
    try:
        cursor = conn.cursor()
        
        # Prepare column names and values
        columns = [
            "document_id", "chapter_number", "section_number", "part_number", "sequence_number",
            "chapter_name", "chapter_tags", "chapter_summary", "chapter_token_count",
            "section_start_page", "section_end_page", "section_importance_score", "section_token_count",
            "section_hierarchy", "section_title", "section_standard", "section_standard_codes", "section_references",
            "content", "embedding"
        ]
        
        # Batch insert using execute_values
        values = []
        for chunk in chunks:
            row = [
                chunk.get("document_id"),
                chunk.get("chapter_number"),
                chunk.get("section_number"),
                chunk.get("part_number"),
                chunk.get("sequence_number"),
                chunk.get("chapter_name"),
                chunk.get("chapter_tags"),
                chunk.get("chapter_summary"),
                chunk.get("chapter_token_count"),
                chunk.get("section_start_page"),
                chunk.get("section_end_page"),
                chunk.get("section_importance_score"),
                chunk.get("section_token_count"),
                chunk.get("section_hierarchy"),
                chunk.get("section_title"),
                chunk.get("section_standard"),
                chunk.get("section_standard_codes"),
                chunk.get("section_references"),
                chunk.get("content"),
                chunk.get("embedding")
            ]
            values.append(row)
        
        # Insert in batches of 100
        batch_size = 100
        for i in range(0, len(values), batch_size):
            batch = values[i:i+batch_size]
            execute_values(
                cursor,
                f"INSERT INTO guidance_sections ({', '.join(columns)}) VALUES %s",
                batch,
                template=None,  # Auto-generate template
                page_size=batch_size
            )
            conn.commit()
            inserted_count += len(batch)
            print(f"Inserted batch: {i//batch_size + 1}/{(len(values) + batch_size - 1)//batch_size}")
        
        print(f"Successfully inserted {inserted_count} chunks.")
        return inserted_count
    
    except Exception as e:
        print(f"Error inserting chunks: {e}")
        traceback.print_exc()
        conn.rollback()
        return 0

def verify_insertion(conn, document_id, expected_count):
    """Verifies that chunks were inserted correctly."""
    try:
        cursor = conn.cursor()
        
        # Check total count
        cursor.execute("SELECT COUNT(*) FROM guidance_sections WHERE document_id = %s", (document_id,))
        actual_count = cursor.fetchone()[0]
        print(f"Verification - Total records: {actual_count} (Expected: {expected_count})")
        
        if actual_count != expected_count:
            print(f"WARNING: Inserted record count {actual_count} does not match expected count {expected_count}")
        
        # Check for null embeddings
        cursor.execute("SELECT COUNT(*) FROM guidance_sections WHERE document_id = %s AND embedding IS NULL", (document_id,))
        null_embeddings = cursor.fetchone()[0]
        print(f"Verification - Records with null embeddings: {null_embeddings}")
        
        if null_embeddings > 0:
            print(f"WARNING: {null_embeddings} records have null embeddings")
        
        # Verify text search vector
        cursor.execute("SELECT COUNT(*) FROM guidance_sections WHERE document_id = %s AND text_search_vector IS NULL", (document_id,))
        null_tsvector = cursor.fetchone()[0]
        print(f"Verification - Records with null text search vectors: {null_tsvector}")
        
        if null_tsvector > 0:
            print(f"WARNING: {null_tsvector} records have null text search vectors")
        
        # Check a random sample of records
        cursor.execute(
            """SELECT id, chapter_number, section_number, part_number, 
               LENGTH(content) AS content_length, 
               array_length(embedding, 1) AS embedding_dimensions
               FROM guidance_sections 
               WHERE document_id = %s 
               ORDER BY RANDOM() LIMIT 5""", 
            (document_id,)
        )
        samples = cursor.fetchall()
        
        print("\nRandom Sample Verification:")
        for sample in samples:
            id, ch, sec, part, content_len, emb_dim = sample
            print(f"ID: {id}, Ch.{ch} Sec.{sec} Part {part}, Content length: {content_len}, Embedding dimensions: {emb_dim}")
        
        return True
    except Exception as e:
        print(f"Error verifying insertion: {e}")
        traceback.print_exc()
        return False

# --- Main Processing Functions ---

def load_chunk_files(input_dir):
    """Loads all chunk files from the input directory."""
    chunks = []
    document_id = None
    
    try:
        json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        print(f"Found {len(json_files)} JSON files in {input_dir}")
        
        if not json_files:
            return [], None
        
        for filename in tqdm(json_files, desc="Loading chunks", unit="file"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
                
                # Get document_id from first file
                if document_id is None:
                    document_id = chunk_data.get("document_id")
                    if not document_id:
                        print(f"Warning: No document_id found in {filename}")
                        document_id = "unknown_document"
                
                # Basic validation
                if "content" not in chunk_data:
                    print(f"Warning: Missing content in {filename}. Skipping.")
                    continue
                
                # Ensure embedding is properly formatted if exists
                if "embedding" in chunk_data and chunk_data["embedding"] is not None:
                    # Verify embedding is a list or array of numbers
                    if not isinstance(chunk_data["embedding"], list):
                        print(f"Warning: Invalid embedding format in {filename}. Skipping.")
                        continue
                    
                    # Check embedding dimensions
                    if len(chunk_data["embedding"]) != 2000:
                        print(f"Warning: Embedding has {len(chunk_data['embedding'])} dimensions instead of 2000 in {filename}")
                
                # Add validated chunk to list
                chunks.append(chunk_data)
        
        print(f"Successfully loaded {len(chunks)} chunks with document_id: {document_id}")
        return chunks, document_id
    
    except Exception as e:
        print(f"Error loading chunk files: {e}")
        traceback.print_exc()
        return [], None

def main():
    """Main execution function."""
    print("-" * 50)
    print("Stage 4: Database Population and Verification")
    print(f"Input directory: {INPUT_DIR}")
    print("-" * 50)
    
    # Load chunk files
    chunks, document_id = load_chunk_files(INPUT_DIR)
    if not chunks or not document_id:
        print("No valid chunks found. Exiting.")
        return
    
    # Connect to database
    conn = create_db_connection()
    if not conn:
        print("Failed to connect to database. Exiting.")
        return
    
    # Setup database schema
    if not setup_database(conn):
        print("Failed to setup database schema. Exiting.")
        conn.close()
        return
    
    # Clear existing data for this document
    if not clear_existing_data(conn, document_id):
        print("Failed to clear existing data. Exiting.")
        conn.close()
        return
    
    # Insert chunks
    inserted_count = insert_chunks(conn, chunks)
    if inserted_count == 0:
        print("Failed to insert any chunks. Exiting.")
        conn.close()
        return
    
    # Verify insertion
    verify_insertion(conn, document_id, len(chunks))
    
    # Close connection
    conn.close()
    print("Database connection closed.")
    
    # Print summary
    print("-" * 50)
    print("Stage 4 Summary:")
    print(f"Document ID: {document_id}")
    print(f"Total chunks processed: {len(chunks)}")
    print(f"Total chunks inserted: {inserted_count}")
    print("-" * 50)
    print("Stage 4 processing complete.")

if __name__ == "__main__":
    main()
