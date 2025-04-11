#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check if pgvector extension is enabled in PostgreSQL database.
"""

import psycopg2
import sys

# Database configuration
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "iris_dev",
    "password": "",  # No password needed for local development
}

def check_pgvector():
    """Connect to the database and check if pgvector extension is enabled."""
    print("Connecting to PostgreSQL database...")
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        
        if result and result[0] == 'vector':
            print("✅ pgvector extension is enabled.")
            
            # Check if vector operators are available
            try:
                cursor.execute("SELECT '[1,2,3]'::vector <=> '[4,5,6]'::vector;")
                print("✅ Vector similarity operator (<=> ) is working properly.")
            except Exception as e:
                print(f"❌ Vector operators not working: {e}")
                
            return True
        else:
            print("❌ pgvector extension is NOT enabled.")
            print("\nTo enable pgvector, run this SQL command:")
            print("CREATE EXTENSION IF NOT EXISTS vector;")
            return False
            
    except Exception as e:
        print(f"Error connecting to database: {e}", file=sys.stderr)
        return False
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    check_pgvector()