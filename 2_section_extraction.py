#!/usr/bin/env python3
"""
Stage 2: Extract sections from chapters and generate section-level details

This script processes chapter JSON files from Stage 1, identifies logical sections,
generates section metadata using GPT, and saves the section-level details.

Input: Chapter JSON files from Stage 1
Output: JSON files with section-level details
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

# --- Configuration ---
INPUT_DIR = "chapter_details"  # Directory with chapter JSON files from Stage 1
OUTPUT_DIR = "section_details"  # Directory to save section-level details
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

def find_headings(text):
    """Find all Markdown headings (H1-H6) with their positions and levels."""
    # Regex for Markdown headings (# to ######)
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+?)\s*(?:\n|$)', re.MULTILINE)
    headings = []
    
    for match in heading_pattern.finditer(text):
        level = len(match.group(1))  # Number of # chars indicates heading level
        title = match.group(2).strip()
        position = match.start()
        headings.append({"level": level, "title": title, "position": position})
    
    return headings

def get_page_range(content, start_pos, end_pos, page_mapping):
    """Determine the page range for a section based on character positions."""
    if not page_mapping:
        return None, None
    
    # Find the page containing the start position
    start_page = None
    for i in range(len(page_mapping) - 1):
        if page_mapping[i][0] <= start_pos < page_mapping[i+1][0]:
            start_page = page_mapping[i][1]
            break
    if start_page is None and start_pos >= page_mapping[-1][0]:
        start_page = page_mapping[-1][1]
    
    # Find the page containing the end position
    end_page = None
    for i in range(len(page_mapping) - 1):
        if page_mapping[i][0] <= end_pos < page_mapping[i+1][0]:
            end_page = page_mapping[i][1]
            break
    if end_page is None and end_pos >= page_mapping[-1][0]:
        end_page = page_mapping[-1][1]
    
    return start_page, end_page

def split_chapter_into_sections(chapter_data):
    """Split a chapter into sections based on headings."""
    raw_content = chapter_data.get("content", "")
    headings = find_headings(raw_content)
    
    if not headings:
        print(f"No headings found in chapter {chapter_data.get('chapter_number')}. Treating entire chapter as one section.")
        return [{
            "section_title": chapter_data.get("chapter_name", "Untitled Section"),
            "section_hierarchy": f"Chapter {chapter_data.get('chapter_number')}",
            "content": raw_content,
            "section_start_position": 0,
            "section_end_position": len(raw_content)
        }]
    
    # Extract page mappings if present in chapter data
    page_mapping = []
    page_number_tag_pattern = re.compile(r'<!--\s*PageNumber="(\d+)"\s*-->')
    for match in page_number_tag_pattern.finditer(raw_content):
        pos = match.start()
        page_num = int(match.group(1))
        page_mapping.append((pos, page_num))
    
    if page_mapping:
        page_mapping.sort(key=lambda x: x[0])
        # Add end of content position with last page number
        page_mapping.append((len(raw_content), page_mapping[-1][1]))
    
    # Create sections from headings
    sections = []
    for i, heading in enumerate(headings):
        # Determine section content
        start_pos = heading["position"]
        end_pos = headings[i+1]["position"] if i < len(headings) - 1 else len(raw_content)
        section_content = raw_content[start_pos:end_pos]
        
        # Create section hierarchy
        hierarchy_parts = []
        current_level = heading["level"]
        hierarchy_parts.append(f"Chapter {chapter_data.get('chapter_number')}")
        
        # For non-chapter headings (below h1), add nested structure
        if current_level > 1:
            matching_higher_headings = []
            for h in reversed(headings[:i]):
                if h["level"] < current_level:
                    matching_higher_headings.insert(0, h)
                    if h["level"] == 1:  # Stop when reaching an h1
                        break
            
            # Add hierarchy parts from higher level headings
            for h in matching_higher_headings:
                hierarchy_parts.append(h["title"])
        
        hierarchy_parts.append(heading["title"])
        section_hierarchy = " > ".join(hierarchy_parts)
        
        # Get page range
        start_page, end_page = get_page_range(raw_content, start_pos, end_pos, page_mapping)
        
        sections.append({
            "section_title": heading["title"],
            "section_hierarchy": section_hierarchy,
            "content": section_content,
            "section_start_position": start_pos,
            "section_end_position": end_pos,
            "section_start_page": start_page if start_page is not None else chapter_data.get("chapter_page_start"),
            "section_end_page": end_page if end_page is not None else chapter_data.get("chapter_page_end")
        })
    
    return sections

def get_section_metadata_from_gpt(section_data, chapter_data, client):
    """Generate metadata for a section using GPT."""
    system_prompt = """
You are an expert in financial reporting and accounting standards analysis. Your task is to generate 
detailed metadata for sections of accounting and financial reporting textbooks.
"""
    
    user_prompt = f"""
Analyze the following section from Chapter {chapter_data.get('chapter_number')} "{chapter_data.get('chapter_name')}". 

Section title: {section_data.get('section_title')}
Section hierarchy: {section_data.get('section_hierarchy')}

Provide the following details in JSON format:

1. "section_title": Confirm or improve the current section title, ensuring it's concise but descriptive
2. "section_standard": Primary accounting standard framework this section addresses (e.g., "IFRS", "US_GAAP", "AASB")
3. "section_standard_codes": List of specific standard references (e.g., ["IFRS 16", "IAS 17"])
4. "section_references": List of cross-references to other sections or standards mentioned
5. "section_importance_score": A float from 0.0 to 1.0 indicating the relative importance of this section for accountants
6. "section_summary": A detailed summary of the section's content and implications

Use the chapter context to inform your analysis:

Chapter summary: {chapter_data.get('chapter_summary')}
Chapter tags: {', '.join(chapter_data.get('chapter_tags', []))}

Here is the section content:

{section_data.get('content')[:45000]}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response_content, usage_info = call_gpt_with_retry(client, messages)
        response_data = json.loads(response_content)
        
        required_fields = ["section_title", "section_standard", "section_standard_codes", 
                          "section_references", "section_importance_score", "section_summary"]
        
        for field in required_fields:
            if field not in response_data:
                print(f"Error: Response missing required field: {field}")
                return None
                
        return response_data
    except Exception as e:
        print(f"Error getting section metadata from GPT: {e}")
        traceback.print_exc()
        return None

# --- Main Processing Functions ---

def process_chapter_sections(chapter_file_path, output_dir, client):
    """Process a chapter file to extract and enhance sections."""
    try:
        # Load chapter data
        with open(chapter_file_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)
        
        chapter_number = chapter_data.get("chapter_number")
        chapter_name = chapter_data.get("chapter_name")
        print(f"Processing sections for Chapter {chapter_number}: {chapter_name}")
        
        # Split chapter into sections
        sections = split_chapter_into_sections(chapter_data)
        print(f"Identified {len(sections)} sections in chapter {chapter_number}")
        
        # Process each section
        processed_sections = []
        for i, section in enumerate(sections):
            print(f"Processing section {i+1}/{len(sections)}: {section.get('section_title')}")
            
            # Calculate token count
            section_content = section.get("content", "")
            section_token_count = count_tokens(section_content)
            
            # Get section metadata from GPT
            section_metadata = get_section_metadata_from_gpt(section, chapter_data, client)
            if not section_metadata:
                print(f"Failed to generate metadata for section {i+1}. Using basic data.")
                section_metadata = {
                    "section_title": section.get("section_title"),
                    "section_standard": "Unknown",
                    "section_standard_codes": [],
                    "section_references": [],
                    "section_importance_score": 0.5,
                    "section_summary": ""
                }
            
            # Create section output data
            section_data = {
                "document_id": chapter_data.get("document_id"),
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
                "section_number": i + 1,  # Sequential section number
                "part_number": 1,  # Default part number (will be updated if split later)
                "sequence_number": None,  # Will be assigned during chunk creation
                
                # Chapter metadata
                "chapter_tags": chapter_data.get("chapter_tags", []),
                "chapter_summary": chapter_data.get("chapter_summary", ""),
                "chapter_token_count": chapter_data.get("chapter_token_count", 0),
                
                # Section structural info
                "section_hierarchy": section.get("section_hierarchy"),
                "section_start_page": section.get("section_start_page"),
                "section_end_page": section.get("section_end_page"),
                "section_token_count": section_token_count,
                
                # Section metadata (from GPT)
                "section_title": section_metadata.get("section_title"),
                "section_standard": section_metadata.get("section_standard"),
                "section_standard_codes": section_metadata.get("section_standard_codes"),
                "section_references": section_metadata.get("section_references"),
                "section_importance_score": section_metadata.get("section_importance_score"),
                "section_summary": section_metadata.get("section_summary"),
                
                # Content
                "content": section_content
            }
            
            processed_sections.append(section_data)
        
        # Save sections to output files
        os.makedirs(output_dir, exist_ok=True)
        output_base = f"{chapter_number:03d}_{chapter_name.replace(' ', '_')}"
        
        for i, section_data in enumerate(processed_sections):
            output_filename = f"{output_base}_section_{i+1:03d}.json"
            output_filepath = os.path.join(output_dir, output_filename)
            
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(section_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved section {i+1} to {output_filename}")
        
        return True, len(processed_sections)
        
    except Exception as e:
        print(f"ERROR processing chapter file {chapter_file_path}: {e}")
        traceback.print_exc()
        return False, 0

def main():
    """Main execution function."""
    print("-" * 50)
    print("Stage 2: Extract Sections and Generate Section-Level Details")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Create OpenAI client
    client = create_openai_client()
    if not client:
        print("Failed to create OpenAI client. Exiting.")
        return
    
    # Find chapter JSON files
    chapter_files = []
    try:
        for file in os.listdir(INPUT_DIR):
            if file.endswith(".json"):
                chapter_files.append(os.path.join(INPUT_DIR, file))
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return
    
    if not chapter_files:
        print(f"No JSON files found in {INPUT_DIR}. Exiting.")
        return
    
    # Sort files naturally by chapter number (extracted from filename)
    def get_chapter_number(filename):
        try:
            return int(os.path.basename(filename).split("_")[0])
        except (ValueError, IndexError):
            return 9999  # Fallback for files with unexpected naming
    
    chapter_files.sort(key=get_chapter_number)
    print(f"Found and sorted {len(chapter_files)} chapter files.")
    
    # Process chapter files
    successful_count = 0
    failed_count = 0
    total_sections = 0
    
    for file_path in tqdm(chapter_files, desc="Processing chapters", unit="file"):
        success, num_sections = process_chapter_sections(file_path, OUTPUT_DIR, client)
        
        if success:
            successful_count += 1
            total_sections += num_sections
        else:
            failed_count += 1
    
    # Print summary
    print("-" * 50)
    print("Stage 2 Summary:")
    print(f"Successfully processed: {successful_count} chapters")
    print(f"Failed to process: {failed_count} chapters")
    print(f"Total sections extracted: {total_sections}")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("-" * 50)
    print("Stage 2 processing complete.")

if __name__ == "__main__":
    main()
