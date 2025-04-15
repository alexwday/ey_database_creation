#!/usr/bin/env python3
"""
Stage 1: Process markdown files into chapter-level details

This script processes markdown files from an input directory, where each file represents
a chapter of the textbook. It extracts metadata, calculates token counts, and generates
chapter-level summaries and tags using GPT.

Input: Markdown files in INPUT_DIR (expected filename format: x_chapter_name.md)
Output: JSON files with chapter-level details in OUTPUT_DIR
"""

import os
import re
import json
import traceback
from pathlib import Path
import tiktoken
import time
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "textbook_chapters"  # Directory containing chapter markdown files
OUTPUT_DIR = "chapter_details"  # Directory to save chapter-level details
DOCUMENT_ID = "YOUR_DOCUMENT_ID_HERE"  # Change this as needed
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4-turbo"  # Or specific model you want to use
MAX_COMPLETION_TOKENS = 2000
TEMPERATURE = 0.5

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

def extract_page_mapping(content):
    """Extracts a mapping of character positions to page numbers from tags."""
    page_number_tag_pattern = re.compile(r'<!--\s*PageNumber="(\d+)"\s*-->')
    mapping = []
    raw_matches = []

    # Find all tags and their positions/page numbers
    for match in page_number_tag_pattern.finditer(content):
        pos = match.start()
        page_num = int(match.group(1))
        raw_matches.append((pos, page_num))

    if not raw_matches:
        return []

    # Create mapping, resolving duplicate positions
    raw_matches.sort(key=lambda x: (x[0], -x[1]))
    mapping.append(raw_matches[0])
    
    for i in range(1, len(raw_matches)):
        if raw_matches[i][0] > mapping[-1][0]:
            mapping.append(raw_matches[i])

    # Ensure mapping covers end of document
    if mapping:
        last_entry_pos, last_entry_page = mapping[-1]
        if last_entry_pos < len(content):
            if not mapping or mapping[-1][0] < len(content):
                mapping.append((len(content), last_entry_page))
            elif mapping[-1][0] == len(content):
                mapping[-1] = (len(content), max(last_entry_page, mapping[-1][1]))

    return mapping

def clean_azure_tags(text):
    """Removes Azure Document Intelligence specific HTML comment tags from text."""
    azure_tag_pattern = re.compile(
        r'<!--\s*Page(Footer|Number|Break|Header)=?(".*?"|\d+)?\s*-->\s*\n?'
    )
    return azure_tag_pattern.sub("", text)

def extract_chapter_info(filename):
    """Extracts chapter number and name from filename."""
    basename = os.path.basename(filename)
    # Try to find chapter number at the beginning of the filename
    chapter_number_match = re.search(r"^(\d+)[_-].*", basename)
    chapter_number = int(chapter_number_match.group(1)) if chapter_number_match else 0
    
    # Extract chapter name by removing number prefix
    name_without_number = re.sub(r"^\d+[_-]", "", os.path.splitext(basename)[0])
    # Replace underscores with spaces and capitalize
    chapter_name = " ".join(word.capitalize() for word in name_without_number.replace("_", " ").split())
    
    return chapter_number, chapter_name

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

def call_gpt_with_retry(client, messages, model=MODEL_NAME, max_tokens=MAX_COMPLETION_TOKENS, 
                        temperature=TEMPERATURE, retries=3, delay=5):
    """Makes API call with retry logic."""
    last_exception = None
    for attempt in range(retries):
        try:
            print(f"Making API call (Attempt {attempt + 1}/{retries})...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            print("API call successful.")
            return response.choices[0].message.content, response.usage
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            last_exception = e
            time.sleep(delay * (attempt + 1))
    
    print(f"API call failed after {retries} attempts.")
    if last_exception:
        raise last_exception
    else:
        raise Exception("API call failed for unknown reasons.")

def get_chapter_metadata_from_gpt(chapter_content, chapter_name, client):
    """Generates summary and tags for a chapter using GPT."""
    system_prompt = """
You are an expert in financial reporting and accounting standards analysis. Your task is to extract 
key information from accounting and financial reporting textbook chapters.
"""
    
    user_prompt = f"""
Please analyze the following chapter titled "{chapter_name}" from an accounting textbook.

Provide a structured summary highlighting the chapter's purpose, key topics covered, 
applicable accounting standards, and the main decisions or judgments accountants would 
make based on this guidance.

Also include a set of specific, relevant tags for efficient retrieval, including relevant 
standard names (e.g., IFRS 15, IAS 36, ASC 606) and core accounting concepts discussed.

Format your response as a JSON object with two fields:
1. "chapter_summary": A detailed structured summary
2. "chapter_tags": A list of 5-15 specific topic tags

Here is the chapter content:

{chapter_content[:75000]}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response_content, usage_info = call_gpt_with_retry(client, messages)
        response_data = json.loads(response_content)
        
        if "chapter_summary" not in response_data or "chapter_tags" not in response_data:
            print("Error: Response missing required fields.")
            return None
            
        return response_data
    except Exception as e:
        print(f"Error getting chapter metadata from GPT: {e}")
        traceback.print_exc()
        return None

# --- Main Processing Functions ---

def process_chapter_file(file_path, output_dir, client, last_processed_end_page=0):
    """Processes a single chapter file and generates metadata."""
    file_basename = os.path.basename(file_path)
    try:
        print(f"Processing chapter file: {file_basename}")
        
        # 1. Extract chapter number and name from filename
        chapter_number, chapter_name_from_filename = extract_chapter_info(file_basename)
        
        # 2. Read the file content
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        # 3. Extract chapter name from first line if present, otherwise use from filename
        content_lines = raw_content.split("\n", 1)
        chapter_name = content_lines[0].strip() if content_lines and content_lines[0].strip() else chapter_name_from_filename
        
        # 4. Extract page mapping
        page_mapping = extract_page_mapping(raw_content)
        
        # 5. Determine chapter page range
        if not page_mapping:
            chapter_page_start = last_processed_end_page + 1
            chapter_page_end = chapter_page_start
            print(f"  WARN: No page tags found. Inferring start page: {chapter_page_start}")
        else:
            chapter_page_start = page_mapping[0][1]  # First page number
            chapter_page_end = page_mapping[-1][1]   # Last page number
            chapter_page_end = max(chapter_page_start, chapter_page_end)  # Ensure end >= start
            print(f"  Pages: {chapter_page_start}-{chapter_page_end}")
        
        # 6. Clean content and calculate token count
        cleaned_content = clean_azure_tags(raw_content)
        chapter_token_count = count_tokens(cleaned_content)
        
        # 7. Generate chapter metadata using GPT
        print(f"  Generating chapter summary and tags with GPT...")
        gpt_metadata = get_chapter_metadata_from_gpt(cleaned_content, chapter_name, client)
        
        if not gpt_metadata:
            print(f"  ERROR: Failed to generate metadata for chapter {chapter_number}")
            return False, 0, 0, None
        
        # 8. Prepare output data
        output_data = {
            "document_id": DOCUMENT_ID,
            "chapter_number": chapter_number,
            "chapter_name": chapter_name,
            "chapter_summary": gpt_metadata.get("chapter_summary", ""),
            "chapter_tags": gpt_metadata.get("chapter_tags", []),
            "chapter_token_count": chapter_token_count,
            "chapter_page_start": chapter_page_start,
            "chapter_page_end": chapter_page_end,
            "content": raw_content,  # Keep raw content for section extraction
            "source_filename": file_basename
        }
        
        # 9. Save to output file
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{chapter_number:03d}_{chapter_name_from_filename}.json"
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Successfully processed chapter {chapter_number}: {chapter_name}")
        print(f"  Tokens: {chapter_token_count}, Tags: {len(gpt_metadata.get('chapter_tags', []))}")
        return True, chapter_page_start, chapter_page_end, output_filename
        
    except Exception as e:
        print(f"ERROR processing file {file_basename}: {e}")
        traceback.print_exc()
        return False, 0, 0, None

def main():
    """Main execution function."""
    print("-" * 50)
    print("Stage 1: Process Markdown Files into Chapter-Level Details")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Create OpenAI client
    client = create_openai_client()
    if not client:
        print("Failed to create OpenAI client. Exiting.")
        return
    
    # Find markdown files
    markdown_files = []
    try:
        for file in os.listdir(INPUT_DIR):
            if file.endswith(".md"):
                markdown_files.append(os.path.join(INPUT_DIR, file))
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return
    
    if not markdown_files:
        print(f"No markdown files found in {INPUT_DIR}. Exiting.")
        return
    
    # Sort files naturally by chapter number in filename
    markdown_files.sort(key=lambda f: extract_chapter_info(f)[0])
    print(f"Found and sorted {len(markdown_files)} markdown files.")
    
    # Process files
    successful_count = 0
    failed_count = 0
    processed_chapters = []
    last_page = 0
    
    for file_path in tqdm(markdown_files, desc="Processing chapters", unit="file"):
        success, start_page, end_page, output_file = process_chapter_file(
            file_path, OUTPUT_DIR, client, last_page
        )
        
        if success:
            successful_count += 1
            last_page = end_page
            if output_file:
                processed_chapters.append((output_file, start_page, end_page))
        else:
            failed_count += 1
    
    # Print summary
    print("-" * 50)
    print("Stage 1 Summary:")
    print(f"Successfully processed: {successful_count} chapters")
    print(f"Failed to process: {failed_count} chapters")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Check for page gaps or overlaps
    if len(processed_chapters) > 1:
        issues = 0
        processed_chapters.sort(key=lambda x: x[0])  # Sort by filename
        
        for i in range(len(processed_chapters) - 1):
            prev_file, _, prev_end = processed_chapters[i]
            curr_file, curr_start, _ = processed_chapters[i + 1]
            
            if curr_start > prev_end + 1:
                # Gap detected
                gap = curr_start - prev_end - 1
                print(f"GAP: {gap} page(s) between '{prev_file}' and '{curr_file}'")
                issues += 1
            elif curr_start <= prev_end:
                # Overlap detected
                overlap = prev_end - curr_start + 1
                print(f"OVERLAP: {overlap} page(s) between '{prev_file}' and '{curr_file}'")
                issues += 1
        
        if issues == 0:
            print("No page gaps or overlaps detected.")
    
    print("Stage 1 processing complete.")

if __name__ == "__main__":
    main()
