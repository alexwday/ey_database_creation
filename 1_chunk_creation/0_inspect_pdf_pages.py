"""
Stage 0.5: Inspect Page Numbers in Markdown Files (Verification Script).

Purpose:
Verifies page numbering within markdown files located in the specified input
directory. Assumes files contain `<!-- PageNumber="X" -->` tags.

Functionality:
1. Scans the input directory for `.md` files.
2. Extracts all `PageNumber` tags from each file.
3. Reports the range of unique page numbers found per file.
4. Displays context around the first `<!-- PageNumber="1" -->` tag if found.

Use Case:
Validates page numbering consistency before subsequent chunking stages.
"""

import os
import re

try:
    import natsort
except ImportError:
    natsort = None  # Handle optional dependency gracefully.

# --- Constants ---
# Regex to find page number tags like <!-- PageNumber="123" -->.
PAGE_NUMBER_TAG_PATTERN = re.compile(r'<!--\s*PageNumber="(\d+)"\s*-->')
# Directory containing the markdown files to inspect.
INPUT_DIR = "1C_mdsplitkit_output"
# Number of characters for context around the Page 1 tag.
CONTEXT_CHARS = 50


# --- Core Logic ---


def extract_page_mapping(content: str) -> list[tuple[int, int]]:
    """
    Extracts a mapping of character positions to page numbers from tags.

    Scans the content for `PageNumber` tags and creates a list of tuples:
    (character_position, page_number). The position marks the start of the tag,
    indicating the page number for the subsequent text. Handles adjacent tags
    and ensures the mapping covers the entire document.

    Args:
        content: The string content of the markdown file.

    Returns:
        A list of (position, page_number) tuples, sorted by position.
    """
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
    # The content from the last tag's position to the end belongs to that tag's page.
    if mapping:
        last_entry_pos, last_entry_page = mapping[-1]
        # Only add end marker if the last tag wasn't already at the very end
        if last_entry_pos < len(content):
            # Check if an entry for len(content) already exists (unlikely but possible)
            if not mapping or mapping[-1][0] < len(content):
                mapping.append((len(content), last_entry_page))
            # If an entry exists at len(content), ensure it has the correct page number
            elif mapping[-1][0] == len(content):
                mapping[-1] = (len(content), max(last_entry_page, mapping[-1][1]))

    # The mapping list is now built correctly, sorted by position,
    # with duplicate positions resolved and the end covered.
    return mapping


def find_page1_tag_context(content: str) -> str | None:
    """
    Finds the first '<!-- PageNumber="1" -->' tag and returns context.

    Args:
        content: The string content of the markdown file.

    Returns:
        A string with the tag and surrounding context (CONTEXT_CHARS before/after),
        or None if not found. Newlines are replaced with '\\n'.
    """
    tag_pattern = r'<!--\s*PageNumber="1"\s*-->'
    match = re.search(tag_pattern, content)
    if match:
        start_pos = match.start()
        end_pos = match.end()
        context_start = max(0, start_pos - CONTEXT_CHARS)
        context_end = min(len(content), end_pos + CONTEXT_CHARS)
        snippet = content[context_start:context_end].replace(
            "\n", "\\n"
        )  # Replace newlines for single-line output
        return f"...{snippet}..."
    return None


def inspect_file_pages(md_file_path: str) -> tuple[list[int] | None, str | None]:
    """
    Inspects a single markdown file for page number tags.

    Reads the file, uses `extract_page_mapping` to find page transitions,
    extracts unique page numbers, and finds context for the first Page 1 tag.

    Args:
        md_file_path: The path to the markdown file.

    Returns:
        A tuple: (sorted list of unique page numbers | None, context string | None).
        Returns (None, None) if an error occurs during file processing.
    """
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Extract the mapping of (character_position, page_number)
        page_mapping = extract_page_mapping(raw_content)  # Uses inlined function

        # Get all page numbers from the mapping
        page_numbers_found = [page for _, page in page_mapping]

        unique_pages = (
            sorted(list(set(page_numbers_found))) if page_numbers_found else []
        )

        page1_context = None
        if 1 in unique_pages:
            page1_context = find_page1_tag_context(raw_content)

        # Return unique pages and context if page 1 found
        return unique_pages, page1_context

    except (IOError, OSError, Exception) as e:  # More specific exceptions + fallback
        # Log error and indicate failure for this file.
        print(f"  ERROR reading or processing {os.path.basename(md_file_path)}: {e}")
        return None, None


def format_page_ranges(page_list: list[int]) -> str:
    """
    Formats a sorted list of integers into a compact range string.

    Example: [1, 2, 3, 5, 7, 8] -> "1-3, 5, 7-8"

    Args:
        page_list: A sorted list of unique page numbers.

    Returns:
        A string representing the page ranges, or "" if the list is empty.
    """
    if not page_list:
        return ""

    page_ranges = []
    start = page_list[0]
    end = start

    for i in range(1, len(page_list)):
        if page_list[i] == end + 1:
            end = page_list[i]
        else:
            if start == end:
                page_ranges.append(str(start))
            else:
                page_ranges.append(f"{start}-{end}")
            start = page_list[i]
            end = start

    # Add the last range
    if start == end:
        page_ranges.append(str(start))
    else:
        page_ranges.append(f"{start}-{end}")

    return ", ".join(page_ranges)


def main():
    """
    Main execution function.

    Finds markdown files in INPUT_DIR, sorts them, inspects each for page
    numbers using `inspect_file_pages`, and prints the results.
    """
    print("-" * 50)
    print("Running Stage 0.5: Inspect Chapter Page Numbers")
    print(f"Input directory: {INPUT_DIR}")
    print("-" * 50)

    # Find all markdown files
    markdown_files = []
    try:
        # Recursively search for .md files
        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if file.endswith(".md"):
                    markdown_files.append(os.path.join(root, file))
    except FileNotFoundError:
        print(f"\nERROR: Input directory not found: {INPUT_DIR}")
        return

    if not markdown_files:
        print(f"\nNo markdown files found in {INPUT_DIR}. Exiting.")
        return

    # Sort markdown files (natural sort if available, otherwise standard sort)
    if natsort:
        markdown_files = natsort.natsorted(markdown_files)
        print(f"Found and naturally sorted {len(markdown_files)} markdown files.")
    else:
        markdown_files.sort()
        print(f"Found {len(markdown_files)} markdown files (standard sort).")
        print(
            "(Optional: Install 'natsort' for potentially better chapter ordering: pip install natsort)"
        )

    print("\nInspecting page numbers in each file:")
    # Process each markdown file found
    for md_file in markdown_files:
        file_basename = os.path.basename(md_file)
        unique_pages, page1_context = inspect_file_pages(md_file)

        if unique_pages is not None:  # Check if processing was successful
            if unique_pages:
                pages_str = format_page_ranges(unique_pages)  # Use helper function
                print(f"  - {file_basename}: Pages Found: {pages_str}")
                if page1_context:
                    print(f"      Context for Page 1 Tag: {page1_context}")
            else:
                print(f"  - {file_basename}: No PageNumber tags found.")

    print("-" * 50)
    print("Inspection complete.")
    print("-" * 50)


if __name__ == "__main__":
    main()
