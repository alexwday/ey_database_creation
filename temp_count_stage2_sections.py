import os
import json
import re
import sys
from pathlib import Path

# --- Configuration ---
# Ensure this points to the correct output file from Stage 1
STAGE1_OUTPUT_DIR = "pipeline_output/stage1"
STAGE1_FILENAME = "stage1_chapter_data.json"

# --- Functions Copied from stage2_section_processing.py ---

def find_headings(raw_content: str) -> list[dict]:
    """Finds Markdown headings (levels 1-6) in raw text."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headings = []
    for match in heading_pattern.finditer(raw_content):
        headings.append({
            "level": len(match.group(1)),
            "text": match.group(2).strip(),
            "position": match.start()
        })
    # Add virtual end marker
    headings.append({"level": 0, "text": "DOCUMENT_END", "position": len(raw_content)})
    headings.sort(key=lambda h: h["position"])
    return headings

def split_chapter_into_sections(chapter_data: dict) -> list[dict]:
    """Splits raw chapter content into initial sections based on headings."""
    # Use 'raw_content' key as expected by stage2 script
    raw_content = chapter_data.get("raw_content", "")
    if not raw_content:
        print(f"Warning: Chapter {chapter_data.get('chapter_number', 'Unknown')} has no 'raw_content'. Skipping.")
        return []

    headings = find_headings(raw_content)
    initial_sections = []
    section_index_in_chapter = 0
    current_heading_context = {f"level_{i}": None for i in range(1, 7)}
    # Initialize L1 with chapter name for context
    current_heading_context["level_1"] = chapter_data.get("chapter_name")

    # Handle content before the first heading
    first_heading_pos = headings[0]['position'] if headings and headings[0]['level'] > 0 else len(raw_content)
    if first_heading_pos > 0:
        intro_slice = raw_content[:first_heading_pos].strip()
        if intro_slice:
            section_index_in_chapter += 1
            # We only need to count, not store full data
            initial_sections.append({"section_number": section_index_in_chapter})

    # Process sections defined by headings
    for i in range(len(headings) - 1):
        current_heading = headings[i]
        next_heading = headings[i + 1]

        # Skip if level is 0 (e.g., the intro section we might have handled)
        if current_heading["level"] == 0: continue

        section_start_pos = current_heading["position"]
        section_end_pos = next_heading["position"]
        raw_section_slice = raw_content[section_start_pos:section_end_pos].strip()

        if raw_section_slice: # Only count section if content exists
            section_index_in_chapter += 1
            # We only need to count, not store full data
            initial_sections.append({"section_number": section_index_in_chapter})

    return initial_sections

# --- Main Counting Logic ---

def count_sections():
    """Loads Stage 1 data and counts total sections identified by the splitting logic."""
    stage1_output_file = Path(STAGE1_OUTPUT_DIR) / STAGE1_FILENAME
    print(f"Attempting to load Stage 1 data from: {stage1_output_file}")

    if not stage1_output_file.exists():
        print(f"ERROR: Stage 1 output file not found: {stage1_output_file}. Cannot proceed.")
        sys.exit(1)

    try:
        with open(stage1_output_file, "r", encoding="utf-8") as f:
            all_chapter_data = json.load(f)
        print(f"Loaded {len(all_chapter_data)} chapters from {stage1_output_file}")
    except Exception as e:
        print(f"ERROR: Error loading Stage 1 data from {stage1_output_file}: {e}")
        sys.exit(1)

    if not all_chapter_data:
        print("WARNING: Stage 1 data is empty. No sections to count.")
        return 0

    total_section_count = 0
    for i, chapter_data in enumerate(all_chapter_data):
        chapter_number = chapter_data.get("chapter_number", f"Unknown (Index {i})")
        print(f"Processing Chapter {chapter_number}...")
        sections = split_chapter_into_sections(chapter_data)
        num_sections = len(sections)
        print(f"  Found {num_sections} sections.")
        total_section_count += num_sections

    print("\n--- Total Section Count ---")
    print(f"Total sections identified across all chapters: {total_section_count}")
    print("--------------------------")

    return total_section_count

if __name__ == "__main__":
    count_sections()
