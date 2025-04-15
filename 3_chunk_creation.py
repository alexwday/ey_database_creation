#!/usr/bin/env python3
"""
Stage 3: Split sections into chunks, merge small chunks, and generate embeddings

This script processes section JSON files from Stage 2, splits large sections,
merges small chunks, and generates embeddings for each chunk.

Input: Section JSON files from Stage 2
Output: Database-ready JSON chunks with embeddings
"""

import os
import re
import json
import traceback
import time
from pathlib import Path
import tiktoken
from openai import OpenAI
from tqdm import tqdm
import numpy as np

# --- Configuration ---
INPUT_DIR = "section_details"  # Directory with section JSON files from Stage 2
OUTPUT_DIR = "database_chunks"  # Directory to save final chunk files
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"  # Embedding model to use
COMPLETION_MODEL = "gpt-4-turbo"  # For any completion tasks

# Token limits for chunking
MAX_CHUNK_TOKENS = 750  # Target maximum tokens per chunk
MIN_CHUNK_TOKENS = 250  # Target minimum tokens per chunk
MERGE_THRESHOLD = 200   # Merge chunks smaller than this

# Initialize tokenizer
try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    print("INFO: Using 'cl100k_base' tokenizer via tiktoken.")
except Exception as e:
    print(f"WARN: Failed to initialize tokenizer: {e}. Falling back to estimate.")
    TOKENIZER = None

# --- Utility Functions ---

def count_tokens(text):
    """Count tokens in text using the tokenizer."""
    if not text:
        return 0
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(text))
        except Exception as e:
            return len(text) // 4  # Fallback
    else:
        return len(text) // 4

def create_openai_client():
    """Creates and returns an OpenAI client."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client created successfully.")
        return client
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        traceback.print_exc()
        return None

def call_with_retry(func, *args, retries=3, delay=5, **kwargs):
    """Generic retry function for API calls."""
    last_exception = None
    for attempt in range(retries):
        try:
            print(f"API call attempt {attempt + 1}/{retries}...")
            result = func(*args, **kwargs)
            print("API call successful.")
            return result
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            last_exception = e
            time.sleep(delay * (attempt + 1))
    
    print(f"API call failed after {retries} attempts.")
    if last_exception:
        raise last_exception
    else:
        raise Exception("API call failed for unknown reasons.")

def generate_embedding(client, text):
    """Generate an embedding for the given text."""
    try:
        return call_with_retry(
            client.embeddings.create,
            input=text,
            model=EMBEDDING_MODEL
        ).data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        traceback.print_exc()
        return None

def split_text_by_tokens(text, max_tokens=MAX_CHUNK_TOKENS):
    """Split text into chunks of max_tokens each, trying to preserve paragraphs and sentences."""
    if not text.strip():
        return []
    
    # Function to check token count of a chunk
    def check_size(text_chunk):
        return count_tokens(text_chunk)
    
    # First split by paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = check_size(para)
        
        # If paragraph is already too big, split by sentences
        if para_size > max_tokens:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
            for sentence in sentences:
                sentence_size = check_size(sentence)
                
                # If sentence is still too big, just add it as its own chunk
                if sentence_size > max_tokens:
                    if current_chunk:  # Save the current chunk first
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    chunks.append(sentence)  # Add big sentence as its own chunk
                # Otherwise, check if adding this sentence would exceed the limit
                elif current_size + sentence_size <= max_tokens:
                    current_chunk.append(sentence)
                    current_size += sentence_size
                else:  # Save current chunk and start a new one
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
        # If paragraph fits within limit
        elif current_size + para_size <= max_tokens:
            current_chunk.append(para)
            current_size += para_size
        else:  # Save current chunk and start a new one with this paragraph
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

def merge_small_chunks(chunks, min_tokens=MIN_CHUNK_TOKENS):
    """Merge adjacent small chunks if they're below the minimum token threshold."""
    if not chunks:
        return []
    
    # If only one chunk, return it regardless of size
    if len(chunks) == 1:
        return chunks
    
    # Check which chunks are candidates for merging
    chunk_sizes = [count_tokens(chunk) for chunk in chunks]
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        # If the current chunk is small and not the last one
        if chunk_sizes[i] < min_tokens and i < len(chunks) - 1:
            # Check if merging with the next chunk would stay under MAX_CHUNK_TOKENS
            combined_size = chunk_sizes[i] + chunk_sizes[i+1]
            if combined_size <= MAX_CHUNK_TOKENS:
                merged_content = chunks[i] + "\n\n" + chunks[i+1]
                merged_chunks.append(merged_content)
                i += 2  # Skip the next chunk since we've merged it
            else:
                # Can't merge, add current chunk as is
                merged_chunks.append(chunks[i])
                i += 1
        else:
            # Current chunk is either big enough or the last one
            merged_chunks.append(chunks[i])
            i += 1
    
    return merged_chunks

# --- Main Processing Functions ---

def process_section_file(section_file_path, output_dir, client):
    """Process a section file: split if needed, merge if needed, generate embeddings."""
    try:
        # Load section data
        with open(section_file_path, "r", encoding="utf-8") as f:
            section_data = json.load(f)
        
        chapter_number = section_data.get("chapter_number")
        section_number = section_data.get("section_number")
        section_title = section_data.get("section_title")
        print(f"Processing Ch.{chapter_number} Sec.{section_number}: {section_title}")
        
        # Check if section needs splitting
        section_content = section_data.get("content", "")
        section_token_count = section_data.get("section_token_count", count_tokens(section_content))
        
        # Process chunks
        chunks = []
        if section_token_count <= MAX_CHUNK_TOKENS:
            # Section is small enough to be a single chunk
            chunks = [section_content]
        else:
            # Split section into chunks
            print(f"  Splitting section {section_number} (tokens: {section_token_count}) into chunks...")
            chunks = split_text_by_tokens(section_content, MAX_CHUNK_TOKENS)
            print(f"  Split into {len(chunks)} initial chunks")
            
            # Check for very small chunks
            small_chunks = [i for i, chunk in enumerate(chunks) if count_tokens(chunk) < MIN_CHUNK_TOKENS]
            if small_chunks and len(chunks) > 1:
                print(f"  Found {len(small_chunks)} small chunks. Merging if possible...")
                chunks = merge_small_chunks(chunks, MIN_CHUNK_TOKENS)
                print(f"  After merging: {len(chunks)} chunks")
        
        # Process each chunk
        processed_chunks = []
        for i, chunk_content in enumerate(chunks):
            # Calculate token count
            chunk_token_count = count_tokens(chunk_content)
            
            # Assign sequence number based on overall position
            sequence_number = (chapter_number * 10000) + (section_number * 100) + (i + 1)
            
            # Create part number for split sections
            part_number = i + 1 if len(chunks) > 1 else 1
            
            # Generate embedding
            print(f"  Generating embedding for chunk {i+1}/{len(chunks)}...")
            embedding = generate_embedding(client, chunk_content)
            
            if embedding is None:
                print(f"  Warning: Failed to generate embedding for chunk {i+1}. Continuing without it.")
            
            # Create chunk output data
            chunk_data = {
                # Structural positioning fields
                "document_id": section_data.get("document_id"),
                "chapter_number": chapter_number,
                "section_number": section_number,
                "part_number": part_number,
                "sequence_number": sequence_number,
                
                # Chapter-level metadata
                "chapter_name": section_data.get("chapter_name"),
                "chapter_tags": section_data.get("chapter_tags", []),
                "chapter_summary": section_data.get("chapter_summary", ""),
                "chapter_token_count": section_data.get("chapter_token_count", 0),
                
                # Section-level pagination & importance
                "section_start_page": section_data.get("section_start_page"),
                "section_end_page": section_data.get("section_end_page"),
                "section_importance_score": section_data.get("section_importance_score", 0.5),
                "section_token_count": section_token_count,
                
                # Section-level metadata
                "section_hierarchy": section_data.get("section_hierarchy"),
                "section_title": section_title,
                "section_standard": section_data.get("section_standard", ""),
                "section_standard_codes": section_data.get("section_standard_codes", []),
                "section_references": section_data.get("section_references", []),
                
                # Content & embedding
                "content": chunk_content,
                "embedding": embedding,
            }
            
            processed_chunks.append(chunk_data)
        
        # Save chunks to output files
        os.makedirs(output_dir, exist_ok=True)
        base_filename = f"{chapter_number:03d}_{section_number:03d}"
        
        for i, chunk_data in enumerate(processed_chunks):
            output_filename = f"{base_filename}_part_{i+1:02d}.json"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Save embedding as regular list for JSON serialization
            if chunk_data["embedding"] is not None:
                chunk_data["embedding"] = list(chunk_data["embedding"])
            
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            print(f"  Saved chunk {i+1}/{len(processed_chunks)} to {output_filename}")
        
        return True, len(processed_chunks)
        
    except Exception as e:
        print(f"ERROR processing section file {section_file_path}: {e}")
        traceback.print_exc()
        return False, 0

def main():
    """Main execution function."""
    print("-" * 50)
    print("Stage 3: Split Sections, Merge Small Chunks, Generate Embeddings")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Create OpenAI client
    client = create_openai_client()
    if not client:
        print("Failed to create OpenAI client. Exiting.")
        return
    
    # Find section JSON files
    section_files = []
    try:
        for file in os.listdir(INPUT_DIR):
            if file.endswith(".json"):
                section_files.append(os.path.join(INPUT_DIR, file))
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return
    
    if not section_files:
        print(f"No JSON files found in {INPUT_DIR}. Exiting.")
        return
    
    # Sort files by chapter and section number
    def get_chapter_section_numbers(filename):
        base = os.path.basename(filename)
        parts = base.split("_")
        try:
            chapter = int(parts[0])
            # Extract section number from the filename
            section_part = [p for p in parts if "section" in p.lower()]
            if section_part:
                section = int(re.search(r'\d+', section_part[0]).group())
            else:
                section = 999  # Fallback
            return (chapter, section)
        except (ValueError, IndexError, AttributeError):
            return (999, 999)  # Fallback for files with unexpected naming
    
    section_files.sort(key=get_chapter_section_numbers)
    print(f"Found and sorted {len(section_files)} section files.")
    
    # Process section files
    successful_count = 0
    failed_count = 0
    total_chunks = 0
    
    for file_path in tqdm(section_files, desc="Processing sections", unit="file"):
        success, num_chunks = process_section_file(file_path, OUTPUT_DIR, client)
        
        if success:
            successful_count += 1
            total_chunks += num_chunks
        else:
            failed_count += 1
    
    # Print summary
    print("-" * 50)
    print("Stage 3 Summary:")
    print(f"Successfully processed: {successful_count} sections")
    print(f"Failed to process: {failed_count} sections")
    print(f"Total chunks created: {total_chunks}")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("-" * 50)
    print("Stage 3 processing complete.")

if __name__ == "__main__":
    main()
