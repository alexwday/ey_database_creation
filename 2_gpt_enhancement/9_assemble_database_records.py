#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3C: Assemble Pre-Embedding Database Records (Notebook Version)

Goal: Combine the chunk data with the generated chapter and section details
into a single list of records matching the target database schema (pre-embedding).
Designed to be run directly in a Jupyter Notebook cell.

Input:
- Chunk JSON files from CHUNK_INPUT_DIR (e.g., 2E_final_merged_chunks).
- Chapter details JSON files from CHAPTER_DETAILS_INPUT_DIR (e.g., 3A_chapter_details).
- Section details JSON files from SECTION_DETAILS_INPUT_DIR (e.g., 3B_section_details).

Output:
- A single JSON file containing a list of all assembled records,
  saved to OUTPUT_FILE (e.g., 3C_pre_embedding_records/pre_embedding_records.json).
"""

import os
import json
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from tqdm.notebook import tqdm # Use tqdm.notebook for Jupyter
import natsort # For sorting section IDs if needed, though order comes from chunks

# --- Configuration ---
# Adjust these paths if your notebook is not in the project root directory
CHUNK_INPUT_DIR = "2E_final_merged_chunks"
CHAPTER_DETAILS_INPUT_DIR = "3A_chapter_details"
SECTION_DETAILS_INPUT_DIR = "3B_section_details"
OUTPUT_DIR = "3C_pre_embedding_records"
OUTPUT_FILENAME = "pre_embedding_records.json"

print("--- Starting Database Record Assembly (Notebook Version) ---")

# --- Helper Functions ---

def sanitize_filename_part(section_id_str):
    """
    Replicates the filename sanitization logic from script 8 for section IDs.
    Needed to correctly find the section detail file.
    """
    # Basic sanitization
    safe_str = str(section_id_str).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').strip()
    # Truncate if too long (matching script 8)
    max_len = 100
    safe_id = (safe_str[:max_len] + '...') if len(safe_str) > max_len else safe_str
    if not safe_id:
        safe_id = "unknown_section" # Fallback
    return safe_id

def load_all_chunks(input_dir):
    """
    Loads all chunk JSONs, validates required fields, and sorts by 'sequence_number'.
    Required fields based on Stage 5 output.
    """
    all_chunks_data = []
    input_path = Path(input_dir)
    print(f"Loading chunks from: {input_path}")

    if not input_path.is_dir():
        print(f"ERROR: Chunk input directory not found: {input_path}")
        return None

    filenames = [f for f in input_path.iterdir() if f.is_file() and f.suffix == ".json"]
    if not filenames:
        print(f"WARNING: No JSON files found in {input_path}")
        return [] # Return empty list if dir exists but no JSONs

    print(f"Found {len(filenames)} chunk files. Loading and validating...")

    loaded_count = 0
    error_count = 0
    skipped_missing_field = 0

    # Fields required from the final chunk file (output of Stage 5)
    required_fields = [
        'sequence_number', 'chapter_number', 'section_number', 'part_number',
        'document_id', 'chapter_name', 'chapter_token_count',
        'section_start_page', 'section_end_page', 'section_token_count',
        'section_hierarchy', 'section_title', 'content', 'chunk_token_count',
        'start_pos', 'end_pos' # Keep start/end pos for potential future use/debugging
    ]

    # Use tqdm.notebook here
    for filepath in tqdm(filenames, desc="Loading Chunks"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- Validation ---
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"WARNING: Missing required fields {missing_fields} in {filepath.name}. Skipping.")
                skipped_missing_field += 1
                continue

            # Basic type checks (add more as needed)
            if not isinstance(data['sequence_number'], int): # Check sequence_number
                 print(f"WARNING: Invalid 'sequence_number' field type in {filepath.name}. Skipping.")
                 skipped_missing_field += 1
                 continue
            if not isinstance(data['section_number'], int): # Check section_number
                 print(f"WARNING: Invalid 'section_number' field type in {filepath.name}. Skipping.")
                 skipped_missing_field += 1
                 continue
            if not isinstance(data['part_number'], int): # Check part_number
                 print(f"WARNING: Invalid 'part_number' field type in {filepath.name}. Skipping.")
                 skipped_missing_field += 1
                 continue
            if not isinstance(data['content'], str):
                 print(f"WARNING: Invalid 'content' field type in {filepath.name}. Skipping.")
                 skipped_missing_field += 1
                 continue
            # --- End Validation ---

            all_chunks_data.append(data)
            loaded_count += 1

        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {filepath.name}. Skipping.")
            error_count += 1
        except Exception as e:
            print(f"ERROR: Error processing file {filepath.name}: {e}. Skipping.")
            # traceback.print_exc() # Optional: uncomment for detailed errors in notebook
            error_count += 1

    print(f"Successfully loaded data for {loaded_count} chunks.")
    if skipped_missing_field > 0:
        print(f"WARNING: Skipped {skipped_missing_field} chunks missing required fields.")
    if error_count > 0:
        print(f"WARNING: Skipped {error_count} chunks due to other errors.")

    if not all_chunks_data:
        print("ERROR: No valid chunks were loaded.")
        return None

    # --- Sort by the 'sequence_number' field ---
    try:
        all_chunks_data.sort(key=lambda x: x['sequence_number'])
        print(f"Successfully sorted {len(all_chunks_data)} chunks by 'sequence_number' field.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during sorting: {e}")
        return None # Cannot proceed without sorting

    return all_chunks_data

def load_chapter_details(input_dir):
    """Loads all chapter detail JSONs into a dictionary keyed by chapter_number."""
    chapter_details_map = {}
    input_path = Path(input_dir)
    print(f"Loading chapter details from: {input_path}")

    if not input_path.is_dir():
        print(f"WARNING: Chapter details directory not found: {input_path}. Chapter details will be missing.")
        return {}

    filenames = [f for f in input_path.iterdir() if f.is_file() and f.name.startswith("chapter_") and f.name.endswith("_details.json")]
    if not filenames:
        print(f"WARNING: No chapter detail files found in {input_path}")
        return {}

    print(f"Found {len(filenames)} chapter detail files. Loading...")

    loaded_count = 0
    error_count = 0

    # Use tqdm.notebook here
    for filepath in tqdm(filenames, desc="Loading Chapter Details"):
        try:
            # Extract chapter number from filename
            parts = filepath.name.split('_')
            if len(parts) < 3 or parts[0] != 'chapter' or parts[-1] != 'details.json':
                print(f"WARNING: Could not parse chapter number from filename: {filepath.name}. Skipping.")
                error_count += 1
                continue
            chapter_num_str = parts[1]
            # Attempt to convert to int if possible, otherwise keep as string
            try:
                chapter_key = int(chapter_num_str)
            except ValueError:
                chapter_key = chapter_num_str

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validation
            if 'chapter_summary' not in data or 'chapter_tags' not in data:
                print(f"WARNING: Chapter details file {filepath.name} is missing 'chapter_summary' or 'chapter_tags'. Skipping.")
                error_count += 1
                continue

            chapter_details_map[chapter_key] = data
            loaded_count += 1

        except json.JSONDecodeError:
            print(f"ERROR: Error decoding JSON from chapter details file: {filepath.name}. Skipping.")
            error_count += 1
        except Exception as e:
            print(f"ERROR: Error loading chapter details file {filepath.name}: {e}")
            error_count += 1

    print(f"Successfully loaded {loaded_count} chapter details.")
    if error_count > 0:
        print(f"WARNING: Failed to load or parse {error_count} chapter detail files.")

    return chapter_details_map

def load_section_details(input_dir):
    """
    Loads all section detail JSONs into a dictionary keyed by (chapter_number, sanitized_section_id).
    Requires parsing chapter and section ID from filenames.
    """
    section_details_map = defaultdict(lambda: None) # Use defaultdict returning None
    input_path = Path(input_dir)
    print(f"Loading section details from: {input_path}")

    if not input_path.is_dir():
        print(f"WARNING: Section details directory not found: {input_path}. Section details will be missing.")
        return section_details_map # Return empty defaultdict

    filenames = [f for f in input_path.iterdir() if f.is_file() and f.name.startswith("chapter_") and f.name.endswith("_details.json") and "_section_" in f.name]
    if not filenames:
        print(f"WARNING: No section detail files found in {input_path}")
        return section_details_map # Return empty defaultdict

    print(f"Found {len(filenames)} potential section detail files. Loading...")

    loaded_count = 0
    error_count = 0
    skipped_parse_error = 0

    required_fields = [
        "section_summary", "section_tags", "section_standard",
        # Use the renamed field from Stage 8
        "section_standard_codes", "section_importance_score", "section_references"
    ]

    temp_section_details_map = {} # Key: (chapter_num, section_number_from_filename)

    # Use tqdm.notebook here
    for filepath in tqdm(filenames, desc="Loading Section Details"):
        try:
            # Extract chapter number and section number from filename
            # Assumes filename format: chapter_{chap_num}_section_{sec_num}_details.json
            filename_stem = filepath.stem # Removes .json
            if not filename_stem.endswith("_details"):
                skipped_parse_error += 1
                continue
            base_name = filename_stem[:-len("_details")] # Remove suffix

            parts = base_name.split('_section_')
            if len(parts) != 2 or not parts[0].startswith("chapter_"):
                print(f"WARNING: Could not parse chapter/section from filename: {filepath.name}. Skipping.")
                skipped_parse_error += 1
                continue

            chapter_part = parts[0]
            section_num_str = parts[1] # This should be the section number
            chapter_num_str = chapter_part[len("chapter_"):]

            # Attempt to convert chapter num to int if possible
            try:
                chapter_key = int(chapter_num_str)
            except ValueError:
                chapter_key = chapter_num_str

            # Attempt to convert section num to int
            try:
                section_key = int(section_num_str)
            except ValueError:
                 print(f"WARNING: Could not parse section number '{section_num_str}' as int from filename: {filepath.name}. Skipping.")
                 skipped_parse_error += 1
                 continue


            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validation
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"WARNING: Section details file {filepath.name} missing fields: {missing_fields}. Skipping.")
                error_count += 1
                continue

            # Type checks (optional but good) - Use print for warnings
            if not isinstance(data.get('section_summary'), str): print(f"WARNING: Type mismatch: 'section_summary' in {filepath.name}")
            if not isinstance(data.get('section_tags'), list): print(f"WARNING: Type mismatch: 'section_tags' in {filepath.name}")
            if not isinstance(data.get('section_standard'), str): print(f"WARNING: Type mismatch: 'section_standard' in {filepath.name}")
            if not isinstance(data.get('section_standard_codes'), list): print(f"WARNING: Type mismatch: 'section_standard_codes' in {filepath.name}")
            # Check renamed field
            if not isinstance(data.get('section_importance_score'), (float, int)): print(f"WARNING: Type mismatch: 'section_importance_score' in {filepath.name}")
            if not isinstance(data.get('section_references'), list): print(f"WARNING: Type mismatch: 'section_references' in {filepath.name}")

            # Key is now (chapter_key, section_key)
            temp_section_details_map[(chapter_key, section_key)] = data
            loaded_count += 1

        except json.JSONDecodeError:
            print(f"ERROR: Error decoding JSON from section details file: {filepath.name}. Skipping.")
            error_count += 1
        except Exception as e:
            print(f"ERROR: Error loading section details file {filepath.name}: {e}")
            error_count += 1

    print(f"Successfully loaded {loaded_count} section details (keyed by sanitized ID).")
    if skipped_parse_error > 0:
        print(f"WARNING: Skipped {skipped_parse_error} files due to filename parsing errors.")
    if error_count > 0:
        print(f"WARNING: Failed to load or parse {error_count} section detail files.")

    return temp_section_details_map # Return map keyed by (chapter_num, section_num)

# Removed build_section_hierarchy function as fields are now directly loaded


# --- Main Assembly Logic (Top Level for Notebook) ---

# 1. Load data
all_chunks = load_all_chunks(CHUNK_INPUT_DIR)
if all_chunks is None:
    print("ERROR: Failed to load chunks. Aborting.")
    # In a notebook, you might raise an error or just stop
    # raise ValueError("Failed to load chunks.")
else:
    chapter_details_map = load_chapter_details(CHAPTER_DETAILS_INPUT_DIR)
    # Keyed by (chapter_num, sanitized_section_id_from_filename)
    temp_section_details_map = load_section_details(SECTION_DETAILS_INPUT_DIR)

    # 2. Create output directory
    output_path = Path(OUTPUT_DIR)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to create output directory {output_path}: {e}")
        all_chunks = None # Prevent further processing

    # 3. Assemble records (only if chunks loaded and output dir created)
    if all_chunks is not None:
        final_records = []
        missing_chapter_details_count = 0
        missing_section_details_count = 0

        print(f"Assembling records for {len(all_chunks)} chunks...")
        # Use tqdm.notebook here
        for chunk in tqdm(all_chunks, desc="Assembling Records"):
            try:
                # --- Get Keys and Data from Chunk ---
                sequence_num = chunk['sequence_number']
                chapter_num = chunk['chapter_number']
                section_num = chunk['section_number'] # Use section_number
                part_num = chunk['part_number']
                doc_id = chunk['document_id']
                chunk_chapter_name = chunk['chapter_name']
                chunk_chapter_tokens = chunk['chapter_token_count']
                chunk_section_start_page = chunk['section_page_start']
                chunk_section_end_page = chunk['section_end_page']
                chunk_section_tokens = chunk['section_token_count']
                chunk_section_hierarchy = chunk['section_hierarchy']
                chunk_section_title = chunk['section_title']
                chunk_content = chunk['content']

                # Convert chapter_num to int if possible for lookup consistency
                try:
                    lookup_chapter_key = int(chapter_num)
                except (ValueError, TypeError):
                    lookup_chapter_key = str(chapter_num) # Keep as string if not int

                # --- Get Chapter Details ---
                chapter_details = chapter_details_map.get(lookup_chapter_key)
                if chapter_details is None:
                    missing_chapter_details_count += 1
                    # Provide defaults if chapter details are missing
                    chapter_summary = ""
                    chapter_tags = []
                else:
                    chapter_summary = chapter_details.get('chapter_summary', "")
                    chapter_tags = chapter_details.get('chapter_tags', [])

                # --- Get Section Details ---
                # Use section_num (already an int from validation) for lookup
                section_lookup_key = (lookup_chapter_key, section_num)
                section_details = temp_section_details_map.get(section_lookup_key)

                if section_details is None:
                    missing_section_details_count += 1
                    # Provide defaults if section details are missing
                    section_summary = "" # Crucial field for DB
                    section_standard = "N/A"
                    section_standard_codes = []
                    section_importance_score = 0.5 # Use renamed field
                    section_references = []
                    section_tags = [] # Added default
                else:
                    section_summary = section_details.get('section_summary', "") # Crucial field for DB
                    section_standard = section_details.get('section_standard', "N/A")
                    section_standard_codes = section_details.get('section_standard_codes', [])
                    section_importance_score = section_details.get('section_importance_score', 0.5) # Use renamed field
                    section_references = section_details.get('section_references', [])
                    section_tags = section_details.get('section_tags', []) # Added default

                # --- Assemble Final Record ---
                record = {
                    # SYSTEM FIELDS (DB Generated)
                    "id": None,
                    "created_at": datetime.now(timezone.utc).isoformat(), # Script generated, may be overwritten by DB

                    # STRUCTURAL POSITIONING FIELDS (From Chunk)
                    "document_id": doc_id,
                    "chapter_number": chapter_num,
                    "section_number": section_num,
                    "part_number": part_num,
                    "sequence_number": sequence_num,

                    # CHAPTER-LEVEL METADATA (From Chunk & Chapter Details)
                    "chapter_name": chunk_chapter_name,
                    "chapter_tags": chapter_tags,
                    "chapter_summary": chapter_summary,
                    "chapter_token_count": chunk_chapter_tokens,

                    # SECTION-LEVEL PAGINATION & IMPORTANCE (From Chunk & Section Details)
                    "section_start_page": chunk_section_start_page,
                    "section_end_page": chunk_section_end_page,
                    "section_importance_score": section_importance_score,
                    "section_token_count": chunk_section_tokens,

                    # SECTION-LEVEL METADATA (From Chunk & Section Details)
                    "section_hierarchy": chunk_section_hierarchy,
                    "section_title": chunk_section_title,
                    "section_standard": section_standard,
                    "section_standard_codes": section_standard_codes,
                    "section_references": section_references,
                    # Note: section_tags from Stage 8 are intermediate, not in final schema

                    # CONTENT & EMBEDDING (Content from Chunk, Embedding later)
                    "content": chunk_content,
                    "embedding": None, # Placeholder
                    "text_search_vector": None, # Placeholder (DB Generated)

                    # INTERMEDIATE FIELD NEEDED IN DB (From Section Details)
                    "section_summary": section_summary
                }

                final_records.append(record)

            except Exception as e:
                print(f"ERROR: Error processing chunk with sequence {chunk.get('sequence_number', 'N/A')}: {e}")
                # traceback.print_exc() # Optional

        print(f"Finished assembling {len(final_records)} records.")
        if missing_chapter_details_count > 0:
            print(f"WARNING: Chapter details were missing for {missing_chapter_details_count} lookups.")
        if missing_section_details_count > 0:
            # Add a note about checking the field names
            print(f"WARNING: Section details were missing for {missing_section_details_count} lookups. Check if 'section_number' in chunks correctly maps to section numbers in section detail filenames.")

        # 4. Save output
        output_file_path = output_path / OUTPUT_FILENAME
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_records, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(final_records)} records to {output_file_path}")
        except Exception as e:
            print(f"ERROR: Failed to save output file {output_file_path}: {e}")

print("--- Database Record Assembly Finished (Notebook Version) ---")
