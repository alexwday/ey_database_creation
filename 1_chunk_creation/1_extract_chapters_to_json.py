"""
Stage 1: Extract Chapters from Markdown to JSON Files.

Purpose:
Processes markdown files (assumed to represent individual chapters or sections)
from an input directory. For each file, it extracts metadata (chapter number,
name, page range), cleans the content by removing specific tags, calculates
an estimated token count, and saves the structured data as a JSON file in an
output directory. It also performs a basic analysis to detect potential page
gaps or overlaps between consecutively processed files based on their derived
page ranges.

Input: Markdown files in `INPUT_DIR` (e.g., from `mdsplitkit`).
Output: JSON files in `OUTPUT_DIR`, one per input markdown file.
"""

import os
import json
import traceback  # Retained for detailed error logging in process_file
import re

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("WARN: 'tiktoken' not installed. Token counts will be estimated (chars/4).")

try:
    import natsort
except ImportError:
    natsort = None  # Optional dependency for natural sorting of filenames.

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Optional dependency for progress bar.


# --- Configuration & Constants ---
INPUT_DIR = "1C_mdsplitkit_output"  # Directory containing input markdown files.
OUTPUT_DIR = "2A_chapter_json"  # Directory to save output JSON files.

# Regex to find page number tags like <!-- PageNumber="123" -->.
PAGE_NUMBER_TAG_PATTERN = re.compile(r'<!--\s*PageNumber="(\d+)"\s*-->')
# Regex to find and remove common Azure Document Intelligence tags and potential extra newline.
AZURE_TAG_PATTERN = re.compile(
    r'<!--\s*Page(Footer|Number|Break|Header)=?(".*?"|\d+)?\s*-->\s*\n?'
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


# --- Utility Functions (Previously Inlined) ---


def clean_azure_tags(text: str) -> str:
    """Removes Azure Document Intelligence specific HTML comment tags from text."""
    return AZURE_TAG_PATTERN.sub("", text)


def extract_page_mapping(content: str) -> list[tuple[int, int]]:
    """
    Extracts a mapping of character positions to page numbers from tags.

    (Identical logic to the function in 0_inspect_pdf_pages.py)
    Scans the content for `PageNumber` tags and creates a list of tuples:
    (character_position, page_number). The position marks the start of the tag,
    indicating the page number for the subsequent text. Handles adjacent tags
    and ensures the mapping covers the entire document.

    Args:
        content: The string content of the markdown file.

    Returns:
        A list of (position, page_number) tuples, sorted by position.
    """
    # --- This function's implementation is identical to the one in ---
    # --- 0_inspect_pdf_pages.py. Consider refactoring into a shared utils module. ---
    mapping = []
    raw_matches = []  # Store raw matches first

    # 1. Find all tags and their positions/page numbers
    for match in PAGE_NUMBER_TAG_PATTERN.finditer(content):
        pos = match.start()
        page_num = int(match.group(1))
        raw_matches.append((pos, page_num))

    if not raw_matches:
        return []  # No tags found

    # 2. Create initial mapping, resolving duplicate positions by keeping higher page number
    if raw_matches:
        # Sort primarily by position, secondarily by page number (desc) to prioritize higher pages at same pos
        raw_matches.sort(key=lambda x: (x[0], -x[1]))

        mapping.append(raw_matches[0])  # Add the first match
        for i in range(1, len(raw_matches)):
            # Add if the position is different from the last added position
            if raw_matches[i][0] > mapping[-1][0]:
                mapping.append(raw_matches[i])
            # If position is the same, the sort already ensured the one with the highest
            # page number came first, so we skip subsequent matches at the same position.

    # 3. Ensure the mapping covers the end of the document
    if mapping:
        last_entry_pos, last_entry_page = mapping[-1]
        if last_entry_pos < len(content):
            if not mapping or mapping[-1][0] < len(content):
                mapping.append((len(content), last_entry_page))
            elif mapping[-1][0] == len(content):
                mapping[-1] = (len(content), max(last_entry_page, mapping[-1][1]))

    return mapping


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the given text.

    Uses the initialized `tiktoken` tokenizer if available. Otherwise, falls
    back to a simple character count estimate (length / 4).

    Args:
        text: The text to count tokens for.

    Returns:
        The estimated or calculated token count.
    """
    if not text:  # Handle empty strings
        return 0
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(text))
        except Exception as e:
            # Fallback if encoding fails for some reason
            print(
                f"WARN: tiktoken encode failed ('{str(e)[:50]}...'). Falling back to estimate."
            )
            return len(text) // 4
    else:
        # Estimate tokens if tokenizer isn't available (approx. 4 chars/token)
        return len(text) // 4


def create_directory(directory: str):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def cleanup_filename(name: str) -> str:
    """
    Cleans a string to make it suitable for use as a filename.

    Removes forbidden characters, replaces whitespace with underscores,
    and truncates to a reasonable length.

    Args:
        name: The original string (e.g., a chapter title).

    Returns:
        A cleaned string safe for filenames.
    """
    # Ensure input is a string
    name = str(name)
    # Remove characters forbidden in many filesystems
    name = re.sub(r'[\\/:?*<>|"\']', "", name).strip()
    # Replace whitespace sequences and underscores with a single underscore
    name = re.sub(r"[\s_]+", "_", name)
    # Truncate to 50 chars and remove leading/trailing underscores
    return name[:50].strip("_")


def extract_chapter_info(filename: str) -> tuple[int, str]:
    """
    Extracts a chapter number and a cleaned chapter name from a filename.

    Assumes filenames might follow patterns like "Chapter_01_Intro.md" or "02-GettingStarted.md".
    Falls back gracefully if patterns aren't matched.

    Args:
        filename: The base name of the file (e.g., "01_Introduction.md").

    Returns:
        A tuple containing: (chapter_number (int), cleaned_chapter_name (str)).
        Defaults chapter number to 0 if not found.
    """
    basename = os.path.basename(filename)
    # Try to find a number preceded by "Chapter_" or followed by "_" or "-"
    chapter_number_match = re.search(
        r"(?:Chapter_)?(\d+)[_ -]", basename, re.IGNORECASE
    )
    chapter_number_str = chapter_number_match.group(1) if chapter_number_match else "0"

    try:
        chapter_number = int(chapter_number_str)
    except ValueError:
        chapter_number = 0  # Default if conversion fails

    # Get the filename without extension
    chapter_name_base = os.path.splitext(basename)[0]
    # Remove the "Chapter_XX_" or "XX_" prefix to get a cleaner name
    cleaned_chapter_name = re.sub(
        r"^(?:Chapter_)?\d+[_ -]+\s*", "", chapter_name_base, flags=re.IGNORECASE
    ).strip()
    # If cleaning resulted in an empty string, use the original base name
    if not cleaned_chapter_name:
        cleaned_chapter_name = chapter_name_base

    return chapter_number, cleaned_chapter_name


# --- Main Processing Logic ---


def process_file(
    md_file_path: str, output_dir: str, last_processed_end_page: int
) -> tuple[bool, int, int, str | None]:
    """
    Processes a single markdown file into a structured JSON output.

    Reads the file, extracts metadata (chapter info, page range), cleans content,
    counts tokens, and saves the result. Determines page range using tags and
    the end page of the previously processed file for context.

    Args:
        md_file_path: Path to the input markdown file.
        output_dir: Directory to save the output JSON file.
        last_processed_end_page: The ending page number of the previously
                                 processed chapter (used for inferring start page).

    Returns:
        A tuple: (success_flag, start_page, end_page, output_filename | None)
    """
    file_basename = os.path.basename(md_file_path)
    try:
        print(f"Processing Stage 1 for: {file_basename}")

        # 1. Extract chapter number and cleaned name from filename
        chapter_number, cleaned_chapter_name = extract_chapter_info(file_basename)

        # 2. Read raw content
        with open(md_file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # 3. Assume first line is title, separate it from main content
        #    (Content stored in JSON will not include this first line)
        content_lines = raw_content.split("\n", 1)
        # chapter_title_line = content_lines[0] # Not used currently
        content_body = content_lines[1] if len(content_lines) > 1 else ""

        # 4. Extract page mapping from the *entire* raw content (including title line)
        #    This ensures page tags right at the start are captured correctly.
        page_mapping = extract_page_mapping(raw_content)

        # 5. Determine chapter page range
        chapter_page_start = 0
        chapter_page_end = 0
        if not page_mapping:
            # If no tags, infer start page based on previous chapter's end.
            chapter_page_start = last_processed_end_page + 1
            chapter_page_end = chapter_page_start  # Assume single page if no tags
            print(
                f"    WARN: No page tags found. Inferring start page: {chapter_page_start}"
            )
        else:
            # Use the page number from the first tag found.
            first_tag_page = page_mapping[0][1]
            # Use the page number from the last tag found.
            # The mapping ensures the last entry corresponds to the highest page number mentioned.
            last_tag_page = page_mapping[-1][1]

            # Determine start page: Use first tag's page number directly.
            # The logic assumes the first tag correctly represents the start page.
            # If the first tag isn't page 1, gap analysis should catch it.
            chapter_page_start = first_tag_page
            chapter_page_end = last_tag_page

            # Ensure end page is not before start page (can happen with unusual tagging)
            chapter_page_end = max(chapter_page_start, chapter_page_end)

            print(
                f"    Tags found. Derived page range: {chapter_page_start}-{chapter_page_end}"
            )

        # 6. Clean the main content body (remove Azure tags) for token calculation
        cleaned_content_body = clean_azure_tags(content_body)

        # 7. Calculate token count on the cleaned main content body
        chapter_tokens = count_tokens(cleaned_content_body)

        # 8. Prepare output data structure
        output_data = {
            "chapter_name": cleaned_chapter_name,  # Use cleaned name from filename
            "chapter_number": chapter_number,
            "content": content_body,  # Store content *without* the first line (assumed title)
            "chapter_tokens": chapter_tokens,
            "chapter_page_start": chapter_page_start,
            "chapter_page_end": chapter_page_end,
            "source_filename": file_basename,  # Original filename for reference
        }

        # 9. Construct output filename (e.g., 001_Introduction.json)
        output_safe_name = cleanup_filename(cleaned_chapter_name)
        output_filename = f"{chapter_number:03d}_{output_safe_name}.json"
        output_filepath = os.path.join(output_dir, output_filename)

        # 10. Save data to JSON file
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(
            f"  Successfully created: {output_filename} (Pages: {chapter_page_start}-{chapter_page_end}, Tokens: {chapter_tokens})"
        )
        return True, chapter_page_start, chapter_page_end, output_filename

    except Exception as e:
        print(f"\nERROR processing file {file_basename} in Stage 1: {e}")
        print("--- Traceback ---")
        print(traceback.format_exc())
        print("--- End Traceback ---")
        return False, 0, 0, None  # Indicate failure


def main():
    """
    Main execution function for Stage 1.

    Finds markdown files, sorts them, processes each using `process_file`,
    saves results to JSON, and performs page gap/overlap analysis.
    """
    print("-" * 50)
    print("Running Stage 1: Create Chapter JSON Files")
    print(f"Input directory : {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Ensure the output directory exists
    create_directory(OUTPUT_DIR)

    # --- Find and Sort Input Files ---
    markdown_files = []
    try:
        # Recursively find all .md files in the input directory
        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if file.endswith(".md"):
                    markdown_files.append(os.path.join(root, file))
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return  # Exit if input directory is missing

    if not markdown_files:
        print(f"No markdown files found in {INPUT_DIR}. Exiting.")
        return  # Exit if no files to process

    # Sort files: Use natural sort if available, otherwise standard sort.
    if natsort:
        markdown_files = natsort.natsorted(markdown_files)
        print(f"Found and naturally sorted {len(markdown_files)} markdown files.")
    else:
        markdown_files.sort()
        print(f"Found {len(markdown_files)} markdown files (standard sort).")
        if natsort is None:  # Only warn if import failed
            print(
                "INFO: Install 'natsort' for potentially better file ordering (pip install natsort)."
            )

    # --- Process Files ---
    successful_files_count = 0
    failed_files_count = 0
    # Store results for gap analysis: (filename, start_page, end_page)
    processed_chapters_info = []
    # Track the end page of the last successfully processed file
    last_processed_end_page = 0

    # Setup progress bar if tqdm is available
    file_iterator = markdown_files
    if tqdm:
        file_iterator = tqdm(
            markdown_files, desc="Stage 1 Processing", unit="file", ncols=100
        )

    # Iterate through sorted markdown files
    for md_file_path in file_iterator:
        success, start_page, end_page, output_filename = process_file(
            md_file_path, OUTPUT_DIR, last_processed_end_page
        )

        if success:
            successful_files_count += 1
            # Update the end page for the next iteration's context
            last_processed_end_page = end_page
            if output_filename:  # Should always be true if success is true
                processed_chapters_info.append((output_filename, start_page, end_page))
        else:
            failed_files_count += 1
            # Don't update last_processed_end_page if processing failed

    # --- Print Summary ---
    print("-" * 50)
    print("Stage 1 Summary:")
    print(f"Successfully processed: {successful_files_count} files")
    print(f"Failed to process    : {failed_files_count} files")
    print(f"Output JSON files are in: {OUTPUT_DIR}")
    print("-" * 50)

    # --- Perform Page Gap/Overlap Analysis ---
    print("Page Gap/Overlap Analysis:")
    if successful_files_count > 1:
        # Sort by filename (which should correspond to chapter order)
        processed_chapters_info.sort(key=lambda x: x[0])

        total_min_page = processed_chapters_info[0][1]
        total_max_page = processed_chapters_info[0][2]
        issues_found = 0

        # Print the first chapter's info
        print(
            f"  File: {processed_chapters_info[0][0]}, Pages: {processed_chapters_info[0][1]}-{processed_chapters_info[0][2]}"
        )

        # Compare each chapter with the previous one
        for i in range(len(processed_chapters_info) - 1):
            prev_file, _, prev_end = processed_chapters_info[i]
            curr_file, curr_start, curr_end = processed_chapters_info[i + 1]

            print(f"  File: {curr_file}, Pages: {curr_start}-{curr_end}")

            # Check for gaps
            if curr_start > prev_end + 1:
                gap = curr_start - prev_end - 1
                print(
                    f"  🚨 GAP DETECTED: {gap} page(s) between '{prev_file}' (ends p.{prev_end}) and '{curr_file}' (starts p.{curr_start})"
                )
                issues_found += 1
            # Check for overlaps or sequence issues (current starts before or at previous end)
            elif curr_start <= prev_end:
                overlap = prev_end - curr_start + 1
                print(
                    f"  ⚠️ OVERLAP/SEQUENCE ISSUE: '{curr_file}' (starts p.{curr_start}) overlaps '{prev_file}' (ends p.{prev_end}) by {overlap} page(s)"
                )
                issues_found += 1

            # Update overall max page
            total_max_page = max(total_max_page, curr_end)

        print("-" * 30)
        print(f"Overall derived page range: {total_min_page} - {total_max_page}")
        if issues_found == 0:
            print("No page gaps or overlaps detected between consecutive files.")
        else:
            print(f"Total page gaps/overlaps detected: {issues_found}")

    elif successful_files_count == 1:
        info = processed_chapters_info[0]
        print(
            f"  Only one file processed ({info[0]}, pages: {info[1]}-{info[2]}). No gaps/overlaps to analyze."
        )
    else:
        print("  No files processed successfully. Cannot perform analysis.")
    print("-" * 50)


if __name__ == "__main__":
    main()
