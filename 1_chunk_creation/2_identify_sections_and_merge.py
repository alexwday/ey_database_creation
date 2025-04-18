"""
Stage 2: Identify Sections, Clean, and Merge Small Sections.

Purpose:
Processes chapter-level JSON files generated by Stage 1. It identifies logical
sections within each chapter based on Markdown headings (H1-H6). Each section's
content is cleaned (removing specific tags), and its token/word counts are
calculated. Sections falling below a minimum token threshold (`MIN_TOKENS`) are
merged with adjacent sections based on hierarchical rules and token limits
(`MAX_TOKENS`) to create more appropriately sized chunks. The final, potentially
merged, sections are saved as individual JSON files.

Input: JSON files from `INPUT_DIR` (Stage 1 output).
Output: JSON files in `OUTPUT_DIR`, one per final section.
"""

import os
import json
import traceback  # Retained for detailed error logging
import re

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("WARN: 'tiktoken' not installed. Token counts will be estimated (chars/4).")

try:
    import natsort
except ImportError:
    natsort = None  # Optional dependency for natural sorting.

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Optional dependency for progress bar.


# --- Configuration & Constants ---
INPUT_DIR = "2A_chapter_json"  # Directory containing Stage 1 chapter JSON files.
OUTPUT_DIR = "2B_merged_sections"  # Directory to save final section JSON files.

# Token count thresholds for merging sections.
OPTIMAL_TOKENS = 500  # Target token count (currently informational).
MAX_TOKENS = 750  # Maximum tokens allowed in a merged section.
MIN_TOKENS = 250  # Sections below this count trigger merging logic (Pass 1).
ULTRA_SMALL_THRESHOLD = (
    25  # Sections below this trigger more aggressive merging (Pass 2).
)

# --- Tokenizer Initialization ---
TOKENIZER = None
if tiktoken:
    try:
        # Use the standard tokenizer for recent OpenAI models.
        TOKENIZER = tiktoken.get_encoding("cl100k_base")
        print("INFO: Using 'cl100k_base' tokenizer via tiktoken.")
    except Exception as e:
        print(
            f"WARN: Failed to initialize tiktoken tokenizer: {e}. Falling back to estimate."
        )
        TOKENIZER = None
else:
    # Warning already printed during import attempt.
    pass

# --- Utility Functions (Mostly Inlined/Duplicated - Consider Refactoring) ---

# Regex to find and remove common Azure Document Intelligence tags.
AZURE_TAG_PATTERN = re.compile(
    r'<!--\s*Page(Footer|Number|Break|Header)=?(".*?"|\d+)?\s*-->\s*\n?'
)


def clean_azure_tags(text: str) -> str:
    """Removes Azure Document Intelligence specific HTML comment tags from text."""
    return AZURE_TAG_PATTERN.sub("", text)


def count_tokens(text: str) -> int:
    """
    Counts tokens using tiktoken if available, otherwise estimates (chars/4).
    (Identical logic to the function in 1_extract_chapters_to_json.py)
    """
    if not text:
        return 0
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(text))
        except Exception as e:
            print(
                f"WARN: tiktoken encode failed ('{str(e)[:50]}...'). Falling back to estimate."
            )
            return len(text) // 4
    else:
        return len(text) // 4


def create_directory(directory: str):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def cleanup_filename(name: str) -> str:
    """
    Cleans a string for use as a filename (removes forbidden chars, replaces spaces).
    (Identical logic to the function in 1_extract_chapters_to_json.py)
    """
    name = str(name)
    name = re.sub(r'[\\/:?*<>|"\']', "", name).strip()
    name = re.sub(r"[\s_]+", "_", name)
    return name[:50].strip("_")


def find_headings(raw_content: str) -> list[dict]:
    """
    Finds all Markdown headings (levels 1-6) in the raw text content.

    Args:
        raw_content: The markdown text to scan.

    Returns:
        A list of dictionaries, each containing 'level', 'text', and 'position'
        of a heading. Includes a virtual 'DOCUMENT_END' heading marker. Sorted by position.
    """
    # Regex to find lines starting with 1 to 6 '#' characters followed by a space.
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headings = []
    for match in heading_pattern.finditer(raw_content):
        level = len(match.group(1))  # Number of '#' gives the level
        text = match.group(2).strip()  # The heading text itself
        position = match.start()  # Character index where the heading starts
        headings.append({"level": level, "text": text, "position": position})

    # Add a virtual heading marker representing the end of the content.
    # This simplifies the section splitting logic later. Level 0 is used.
    headings.append({"level": 0, "text": "DOCUMENT_END", "position": len(raw_content)})

    # Ensure headings are sorted by their position in the text.
    headings.sort(key=lambda h: h["position"])
    return headings


def generate_hierarchy_string(section_data: dict) -> str:
    """Generates a breadcrumb-style hierarchy string from level_X fields."""
    parts = []
    # Check level_1 up to the section's own level (or max 6)
    max_level_to_check = section_data.get("level", 6)
    for i in range(1, max_level_to_check + 1):
        level_key = f"level_{i}"
        heading_text = section_data.get(level_key)
        if heading_text:
            parts.append(heading_text)
        else:
            # Stop if a level is missing in the hierarchy path
            break
    return " > ".join(parts)


def get_heading_for_filename(section_data: dict) -> str:
    """
    Retrieves the most specific heading text for use in filenames.
    Checks `level_6` down to `level_1`. Uses section_title as fallback.

    Args:
        section_data: The dictionary representing a section, containing keys like
                      'level_1', 'level_2', ..., 'chapter_name'.

    Returns:
        The most specific heading text found, or 'Untitled_Section' as a fallback.
    """
    # Check from the most specific level down to level 1
    current_level = section_data.get("level", 6) # Start from section's level or max 6
    for level_num in range(current_level, 0, -1):
        level_key = f"level_{level_num}"
        heading_text = section_data.get(level_key)
        if heading_text:
            return heading_text

    # Fallback to the section_title if no level_X heading found
    return section_data.get("section_title", "Untitled_Section")


# --- Core Sectioning and Merging Logic ---


def split_chapter_into_sections(chapter_data: dict, headings: list[dict]) -> list[dict]:
    """
    Splits raw chapter content into initial sections based on heading levels (1-6).

    Each section starts at a heading and ends just before the next heading. Metadata
    about the heading hierarchy and pass-through fields are added.

    Args:
        chapter_data: Dictionary containing chapter info from Stage 1 JSON
                      (must include 'content', 'chapter_name', 'chapter_number',
                       'document_id', 'chapter_token_count', 'chapter_page_start',
                       'chapter_page_end').
        headings: List of heading dicts from `find_headings` (must include DOCUMENT_END).

    Returns:
        A list of dictionaries, each representing a raw section including its
        content slice, position, and hierarchical heading context.
    """
    raw_content = chapter_data["content"]
    chapter_name = chapter_data["chapter_name"]
    chapter_number = chapter_data["chapter_number"]
    document_id = chapter_data["document_id"]
    chapter_token_count = chapter_data["chapter_token_count"]
    chapter_page_start = chapter_data["chapter_page_start"]
    chapter_page_end = chapter_data["chapter_page_end"]

    # If no headings found (besides DOCUMENT_END), treat the whole chapter as one section.
    if len(headings) <= 1:
        section_title = chapter_name # Use chapter name as title if no headings
        return [
            {
                "document_id": document_id,
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
                "chapter_token_count": chapter_token_count,
                "chapter_page_start": chapter_page_start,
                "chapter_page_end": chapter_page_end,
                "raw_content_slice": raw_content,
                "level": 1,
                "level_1": chapter_name,
                "section_title": section_title,
                "start_pos": 0,
                "end_pos": len(raw_content),
                "section_number": 1, # First section
            }
        ]

    # Stores the current active heading text for each level (1-6)
    current_heading_context = {f"level_{i}": None for i in range(1, 7)}
    current_heading_context["level_1"] = chapter_name # Initialize L1 with chapter name

    sections = []
    section_index_in_chapter = 0  # Track section order within the chapter

    # Iterate through headings to define section boundaries
    for i in range(len(headings) - 1):  # Stop before the DOCUMENT_END marker
        current_heading = headings[i]
        next_heading = headings[i + 1]

        # Define the start and end character positions for this section's content
        section_start_pos = current_heading["position"]
        section_end_pos = next_heading["position"]

        # Skip if the section would be empty (e.g., consecutive headings)
        if section_start_pos >= section_end_pos:
            continue

        # Update the heading context based on the current heading's level
        current_heading_info = current_heading
        current_level = current_heading_info["level"]
        current_title = current_heading_info["text"]

        if 1 <= current_level <= 6:
            current_heading_context[f"level_{current_level}"] = current_title
            # Reset lower-level headings in the context
            for lower_level in range(current_level + 1, 7):
                current_heading_context[f"level_{lower_level}"] = None
        else: # Handle potential intro section before first heading
             current_title = chapter_name # Use chapter name if no heading

        # Extract the raw text slice for this section
        raw_section_slice = raw_content[section_start_pos:section_end_pos]

        # Only create a section if its content is not just whitespace
        if raw_section_slice.strip():
            section_index_in_chapter += 1
            section_data = {
                # Pass-through fields
                "document_id": document_id,
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
                "chapter_token_count": chapter_token_count,
                "chapter_page_start": chapter_page_start,
                "chapter_page_end": chapter_page_end,
                # Section specific fields
                "raw_content_slice": raw_section_slice,
                "level": current_level if current_level > 0 else 1, # Assign level, default 1
                "section_title": current_title, # Heading text starting this section
                "start_pos": section_start_pos,
                "end_pos": section_end_pos,
                "section_number": section_index_in_chapter, # Renamed from orig_section_num
            }
            # Add the current heading context (level_1 to level_6)
            for level_num in range(1, 7):
                level_key = f"level_{level_num}"
                if current_heading_context.get(level_key):
                    section_data[level_key] = current_heading_context[level_key]

            sections.append(section_data)

    return sections


def merge_small_sections(
    sections: list[dict], min_tokens: int, max_tokens: int, ultra_small_threshold: int
) -> list[dict]:
    """
    Merges sections smaller than `min_tokens` or `ultra_small_threshold`
    with adjacent sections, respecting hierarchy and `max_tokens` limit.

    Operates in two passes:
    1. Merges sections < `min_tokens` based on level and proximity.
    2. Merges sections < `ultra_small_threshold` more aggressively.

    Args:
        sections: List of cleaned section dictionaries (must include 'content',
                  'section_token_count', 'start_pos', 'end_pos', 'level', 'chapter_number',
                  and other pass-through fields).
        min_tokens: Threshold for the first merging pass.
        max_tokens: Maximum allowed tokens for a merged section.
        ultra_small_threshold: Threshold for the second, more aggressive pass.

    Returns:
        A list of sections after merging potentially small ones.
    """
    if not sections:
        return []

    # Ensure sections are sorted by their original position (section_number) before merging
    sections_to_process = sorted(sections, key=lambda s: s["section_number"])

    # --- Pass 1: Merge sections smaller than `min_tokens` ---
    pass1_merged = []
    i = 0
    while i < len(sections_to_process):
        current = sections_to_process[i]
        # Use section_token_count for merging decisions
        current_tokens = current.get("section_token_count", 0)

        # Keep sections that are already large enough
        if current_tokens >= min_tokens:
            pass1_merged.append(current)
            i += 1
            continue

        # Attempt to merge small sections
        merged_pass1 = False

        # Strategy 1: Merge forward with next section if compatible levels and combined size <= max_tokens
        if i + 1 < len(sections_to_process):
            next_s = sections_to_process[i + 1]
            next_tokens = next_s.get("section_token_count", 0)
            if (
                current["chapter_number"] == next_s["chapter_number"]
                and current.get("level") == next_s.get("level") # Require same level for forward merge
                and current_tokens + next_tokens <= max_tokens
            ):
                # Create merged section data, keeping metadata from the *first* section (current)
                merged_data = current.copy() # Start with current's metadata
                merged_data["content"] = f"{current.get('content', '')}\n\n{next_s.get('content', '')}"
                # Recalculate token count for the merged content
                merged_data["section_token_count"] = count_tokens(merged_data["content"])
                # Keep word count for reference, though less critical now
                merged_data["word_count"] = current.get("word_count", 0) + next_s.get("word_count", 0)
                merged_data["end_pos"] = next_s["end_pos"]
                # All other fields (hierarchy, pass-through) are inherited from 'current'

                pass1_merged.append(merged_data)
                i += 2  # Skip the next section as it's now merged
                merged_pass1 = True

        # Strategy 2: Merge backward with the previously added section if compatible
        if not merged_pass1 and pass1_merged:
            prev_s = pass1_merged[-1]
            prev_tokens = prev_s.get("section_token_count", 0)
            # Check chapter, token limits, and compatible levels (current is same or deeper level)
            if (
                current["chapter_number"] == prev_s["chapter_number"]
                and prev_tokens + current_tokens <= max_tokens
                and current.get("level", 1) >= prev_s.get("level", 1) # Allow merging deeper level back
            ):
                # Merge current's content into the previous section
                prev_s["content"] = f"{prev_s.get('content', '')}\n\n{current.get('content', '')}"
                # Recalculate token count for the merged content
                prev_s["section_token_count"] = count_tokens(prev_s["content"])
                prev_s["word_count"] = prev_s.get("word_count", 0) + current.get("word_count", 0)
                prev_s["end_pos"] = current["end_pos"]
                # Metadata (hierarchy, pass-through) remains from prev_s

                i += 1 # Move to the next section to process
                merged_pass1 = True

        # If no merge occurred, keep the current section as is
        if not merged_pass1:
            pass1_merged.append(current)
            i += 1

    # --- Pass 2: Merge "ultra-small" sections (< ultra_small_threshold) ---
    if not pass1_merged:  # Skip if Pass 1 resulted in nothing
        return []

    final_merged = []
    i = 0
    while i < len(pass1_merged):
        current = pass1_merged[i]
        # Use section_token_count for decisions here too
        current_tokens = current.get("section_token_count", 0)

        # Keep sections that meet the ultra-small threshold
        if current_tokens >= ultra_small_threshold:
            final_merged.append(current)
            i += 1
            continue

        # Determine if the section content looks like just a heading
        is_heading_only = (
            re.match(r"^\s*#+\s+[^#]", current.get("content", "").strip()) is not None
        )
        merged_pass2 = False

        # --- Preferred Merge Direction ---
        if is_heading_only:
            # Headings prefer merging FORWARD (merge *current* heading into the *next* section's content)
            if i + 1 < len(pass1_merged):
                next_s = pass1_merged[i + 1]
                next_tokens = next_s.get("section_token_count", 0)
                if (
                    current["chapter_number"] == next_s["chapter_number"]
                    and current_tokens + next_tokens <= max_tokens
                ):
                    # Create merged section, taking metadata from NEXT section but content from both
                    merged_data = next_s.copy() # Start with next section's metadata
                    merged_data["content"] = f"{current.get('content', '')}\n\n{next_s.get('content', '')}"
                    merged_data["section_token_count"] = count_tokens(merged_data["content"])
                    merged_data["word_count"] = current.get("word_count", 0) + next_s.get("word_count", 0)
                    merged_data["start_pos"] = current["start_pos"] # Start pos from current
                    # Hierarchy etc. comes from next_s

                    final_merged.append(merged_data)
                    i += 2 # Skip current and next
                    merged_pass2 = True
        else:
            # Content sections prefer merging BACKWARD (merge *current* content into the *previous* section)
            if final_merged:
                prev_s = final_merged[-1]
                prev_tokens = prev_s.get("section_token_count", 0)
                if (
                    current["chapter_number"] == prev_s["chapter_number"]
                    and prev_tokens + current_tokens <= max_tokens
                ):
                    # Merge current's content into previous
                    prev_s["content"] = f"{prev_s.get('content', '')}\n\n{current.get('content', '')}"
                    prev_s["section_token_count"] = count_tokens(prev_s["content"])
                    prev_s["word_count"] = prev_s.get("word_count", 0) + current.get("word_count", 0)
                    prev_s["end_pos"] = current["end_pos"]
                    # Metadata remains from prev_s
                    i += 1 # Move to next item in pass1_merged
                    merged_pass2 = True

        # --- Fallback Merge Direction (if preferred failed) ---
        if not merged_pass2:
            if is_heading_only:
                # Fallback for heading: Merge BACKWARD (merge *current* heading into *previous*)
                if final_merged:
                    prev_s = final_merged[-1]
                    prev_tokens = prev_s.get("section_token_count", 0)
                    if (
                        current["chapter_number"] == prev_s["chapter_number"]
                        and prev_tokens + current_tokens <= max_tokens
                    ):
                        # Merge current's content into previous
                        prev_s["content"] = f"{prev_s.get('content', '')}\n\n{current.get('content', '')}"
                        prev_s["section_token_count"] = count_tokens(prev_s["content"])
                        prev_s["word_count"] = prev_s.get("word_count", 0) + current.get("word_count", 0)
                        prev_s["end_pos"] = current["end_pos"]
                        # Metadata remains from prev_s
                        i += 1 # Move to next item in pass1_merged
                        merged_pass2 = True
            else:
                # Fallback for content: Merge FORWARD (merge *current* content into *next*)
                if i + 1 < len(pass1_merged):
                    next_s = pass1_merged[i + 1]
                    next_tokens = next_s.get("section_token_count", 0)
                    if (
                        current["chapter_number"] == next_s["chapter_number"]
                        and current_tokens + next_tokens <= max_tokens
                    ):
                        # Create merged section, taking metadata from NEXT section
                        merged_data = next_s.copy() # Start with next section's metadata
                        merged_data["content"] = f"{current.get('content', '')}\n\n{next_s.get('content', '')}"
                        merged_data["section_token_count"] = count_tokens(merged_data["content"])
                        merged_data["word_count"] = current.get("word_count", 0) + next_s.get("word_count", 0)
                        merged_data["start_pos"] = current["start_pos"] # Start pos from current
                        # Hierarchy etc. comes from next_s

                        final_merged.append(merged_data)
                        i += 2 # Skip current and next
                        merged_pass2 = True

        # If no merge happened in Pass 2 either, keep the ultra-small section
        if not merged_pass2:
            final_merged.append(current)
            i += 1

    return final_merged


def process_chapter_json(chapter_json_path: str, output_dir: str) -> int:
    """
    Processes a single chapter JSON file: splits into sections, cleans,
    merges small sections, and saves the final sections to individual JSON files.

    Args:
        chapter_json_path: Path to the input chapter JSON file (from Stage 1).
        output_dir: Directory to save the output section JSON files.

    Returns:
        The number of final sections saved for this chapter, or 0 on error.
    """
    final_saved_count = 0
    chapter_file_basename = os.path.basename(chapter_json_path)
    try:
        print(f"Processing Stage 2 for: {chapter_file_basename}")

        # 1. Load chapter data from Stage 1 JSON
        with open(chapter_json_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)

        # Extract required fields (including new pass-through fields)
        raw_content = chapter_data["content"] # Now includes title line
        chapter_number = chapter_data["chapter_number"]
        source_md_filename = chapter_data["source_filename"] # Original MD filename

        # 2. Find headings in the raw content
        headings = find_headings(raw_content)

        # 3. Split into initial sections based on headings, passing chapter_data
        initial_sections = split_chapter_into_sections(chapter_data, headings)
        print(f"  Initial sections identified: {len(initial_sections)}")

        # 4. Clean content and calculate initial metrics for each section
        cleaned_sections = []
        for section_raw in initial_sections:
            # Clean Azure tags from the raw slice
            cleaned_content = clean_azure_tags(section_raw["raw_content_slice"])
            # Only keep sections with non-whitespace content after cleaning
            if cleaned_content.strip():
                section_clean = section_raw.copy() # Keep all fields from initial split
                section_clean["content"] = cleaned_content # Store cleaned content
                # Calculate and store the *initial* token count
                section_clean["section_token_count"] = count_tokens(cleaned_content)
                # Keep word count for reference
                section_clean["word_count"] = len(re.findall(r"\w+", cleaned_content))
                # Remove raw slice now that we have cleaned content
                section_clean.pop("raw_content_slice", None)
                cleaned_sections.append(section_clean)
        print(f"  Sections after cleaning & filtering empty: {len(cleaned_sections)}")

        # 5. Merge small sections using the defined thresholds
        # The merge function now uses 'section_token_count' for decisions and updates it
        merged_sections = merge_small_sections(
            cleaned_sections, MIN_TOKENS, MAX_TOKENS, ULTRA_SMALL_THRESHOLD
        )
        print(f"  Sections after merging small ones: {len(merged_sections)}")

        # 6. Save final sections to individual JSON files
        section_output_index = 0 # Index for output filename within this chapter
        for section_data in merged_sections:
            section_output_index += 1

            # Prepare data for saving, ensuring all required fields are present
            save_data = section_data.copy()

            # Rename the final token count field to 'chunk_token_count'
            if "section_token_count" in save_data:
                 save_data["chunk_token_count"] = save_data.pop("section_token_count")
            else:
                 # Should not happen if merge logic is correct, but fallback
                 save_data["chunk_token_count"] = count_tokens(save_data.get("content", ""))

            # Generate the hierarchy string
            save_data["section_hierarchy"] = generate_hierarchy_string(save_data)

            # Determine filename based on most specific heading
            heading_for_filename = get_heading_for_filename(save_data)
            clean_heading_name = cleanup_filename(heading_for_filename)
            # Fallback filename part if cleaning results in empty string
            if not clean_heading_name:
                clean_heading_name = f"Section{save_data.get('level', 0)}"

            # Construct output filename: ChapterNum_SectionIndex_HeadingName.json
            output_filename = f"{chapter_number:03d}_{section_output_index:03d}_{clean_heading_name}.json"
            output_filepath = os.path.join(output_dir, output_filename)

            # Add source file references to the output data
            save_data["source_chapter_json_file"] = chapter_file_basename
            save_data["source_markdown_file"] = source_md_filename

            # Remove intermediate fields not needed in final output (optional cleanup)
            save_data.pop("word_count", None)
            # Keep level_X fields for now as they define hierarchy

            # Write the section data to its JSON file
            try:
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                final_saved_count += 1
            except Exception as write_err:
                 print(f"\nERROR writing file {output_filename}: {write_err}")
                 # Continue processing other sections

        print(f"  Saved {final_saved_count} final sections for this chapter.")
        return final_saved_count

    except Exception as e:
        print(f"\nERROR processing chapter {chapter_file_basename} in Stage 2: {e}")
        print("--- Traceback ---")
        print(traceback.format_exc())
        print("--- End Traceback ---")
        return 0  # Indicate failure for this chapter


def main():
    """
    Main execution function for Stage 2.

    Finds chapter JSON files, sorts them, processes each using
    `process_chapter_json` for sectioning and merging, and saves the
    resulting sections to the output directory.
    """
    print("-" * 50)
    print("Running Stage 2: Sectioning and Merging")
    print(f"Input directory : {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Ensure output directory exists
    create_directory(OUTPUT_DIR)

    # --- Find and Sort Input Files ---
    chapter_json_files = []
    try:
        # List all files in the input directory
        all_files = os.listdir(INPUT_DIR)
        # Filter for JSON files
        chapter_json_files = [
            os.path.join(INPUT_DIR, f) for f in all_files if f.endswith(".json")
        ]
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return  # Exit if input directory is missing

    if not chapter_json_files:
        print(f"No chapter JSON files found in {INPUT_DIR}. Exiting.")
        return  # Exit if no files to process

    # Sort files: Use natural sort if available for better chapter order.
    if natsort:
        chapter_json_files = natsort.natsorted(chapter_json_files)
        print(
            f"Found and naturally sorted {len(chapter_json_files)} chapter JSON files."
        )
    else:
        chapter_json_files.sort()  # Standard sort as fallback
        print(f"Found {len(chapter_json_files)} chapter JSON files (standard sort).")
        if natsort is None:  # Only warn if import failed
            print(
                "INFO: Install 'natsort' for potentially better file ordering (pip install natsort)."
            )

    # --- Process Files ---
    total_sections_saved = 0
    processed_chapter_count = 0
    failed_chapter_count = 0

    # Setup progress bar if tqdm is available
    file_iterator = chapter_json_files
    if tqdm:
        file_iterator = tqdm(
            chapter_json_files, desc="Stage 2 Processing", unit="chapter", ncols=100
        )

    # Process each chapter JSON file
    for chapter_file_path in file_iterator:
        sections_saved_count = process_chapter_json(chapter_file_path, OUTPUT_DIR)
        if sections_saved_count > 0:
            total_sections_saved += sections_saved_count
            processed_chapter_count += 1
        else:
            # If process_chapter_json returned 0, it indicates an error during processing for that chapter.
            failed_chapter_count += 1

    # --- Print Summary ---
    print("-" * 50)
    print("Stage 2 Summary:")
    print(f"Chapters processed successfully: {processed_chapter_count}")
    print(f"Chapters failed processing   : {failed_chapter_count}")
    print(f"Total final sections saved   : {total_sections_saved}")
    print(f"Output section JSON files are in: {OUTPUT_DIR}")
    print("-" * 50)


if __name__ == "__main__":
    main()
