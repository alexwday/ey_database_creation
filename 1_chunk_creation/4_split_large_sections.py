"""
Stage 4: Split Large Sections, Deduplicate, and Finalize Chunks.

Purpose:
Processes paged section JSON files from Stage 3. For each section:
1. Checks if its token count exceeds `MAX_TOKENS`.
2. If within the limit, saves the section directly as a single final chunk JSON file (part 0).
3. If exceeding the limit:
    a. Retrieves the original chapter markdown content.
    b. Extracts tag positions from the original markdown for position mapping.
    c. Splits the section's *cleaned* content into smaller sub-chunks, prioritizing
       paragraph boundaries, then sentence boundaries if paragraphs are too large.
    d. Calculates the start/end character positions of each sub-chunk relative to the
       *cleaned* section content.
    e. Maps these 'clean' positions back to their corresponding positions in the
       *original raw* chapter markdown content, accounting for removed tags.
    f. Calculates a hash of each sub-chunk's content to identify and discard duplicates
       arising from the splitting process within the *same original section*.
    g. Saves each unique sub-chunk as a final chunk JSON file, updating metadata:
       - `start_pos`/`end_pos` reflect positions in the original raw markdown.
       - `chunk_token_count` stores the sub-chunk's token count.
       - `chunk_part_number` indicates the sequence (1, 2, 3...) if split.
       - Appends "(Part X)" to the most specific heading for clarity.

Input: Paged section JSON files from `INPUT_DIR` (Stage 3 output).
       Original chapter markdown files in `ORIGINAL_MD_DIR`.
Output: Final chunk JSON files in `OUTPUT_DIR`.
"""

import os
import json
import traceback  # Retained for detailed error logging
import re
import hashlib  # Used for content deduplication

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
INPUT_DIR = (
    "2C_paged_sections"  # Directory containing Stage 3 paged section JSON files.
)
OUTPUT_DIR = "2D_final_chunks"  # Directory to save the final chunk JSON files.
ORIGINAL_MD_DIR = (
    "1C_mdsplitkit_output"  # Directory containing the original source markdown files.
)

# Token count thresholds. MAX_TOKENS is the primary trigger for splitting.
OPTIMAL_TOKENS = 500  # Target token count (informational).
MAX_TOKENS = 750  # Sections exceeding this are split.
MIN_TOKENS = 250  # Minimum size considered in previous stages (informational here).

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


def count_tokens(text: str) -> int:
    """
    Counts tokens using tiktoken if available, otherwise estimates (chars/4).
    (Identical logic to previous stages)
    """
    if not text:
        return 0
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(text))
        except Exception as e:
            # Fallback if encoding fails
            # print(f"WARN: tiktoken encode failed ('{str(e)[:50]}...'). Falling back to estimate.") # Keep commented out for prod
            return len(text) // 4
    else:
        # Estimate tokens if tokenizer isn't available
        return len(text) // 4


def extract_tag_mapping(raw_content: str) -> list[dict]:
    """
    Finds all Azure Document Intelligence tags in the raw content and returns
    their start position, end position, and length.

    Args:
        raw_content: The original text content possibly containing tags.

    Returns:
        A list of dictionaries, each representing a tag with 'start', 'end',
        and 'length' keys.
    """
    tag_mapping = []
    for match in AZURE_TAG_PATTERN.finditer(raw_content):
        tag_mapping.append(
            {
                "start": match.start(),
                "end": match.end(),
                "length": match.end() - match.start(),
            }
        )
    return tag_mapping


def map_clean_to_raw_pos(
    clean_pos: int, section_raw_start: int, section_tag_mapping: list[dict]
) -> int:
    """
    Maps a character position from the *cleaned* section content back to its
    corresponding position in the *original raw* chapter content.

    It accounts for the characters removed by `clean_azure_tags` by iterating
    through the tag positions relevant to the section.

    Args:
        clean_pos: The character position within the cleaned section content
                   (relative to the start of the cleaned section, i.e., starts at 0).
        section_raw_start: The starting character position of this section in the
                           *original raw* chapter content.
        section_tag_mapping: A list of tag mappings (dicts with 'start', 'end', 'length')
                             that were present *within* the raw slice corresponding
                             to this section. Tag positions should be relative to the
                             start of the raw chapter content.

    Returns:
        The corresponding character position in the original raw chapter content.
    """
    accumulated_tag_length = 0
    # Sort tags by their start position in the raw content
    section_tag_mapping.sort(key=lambda t: t["start"])

    # Iterate through tags that were within the original raw section slice
    for tag in section_tag_mapping:
        # Calculate the tag's start position relative to the *cleaned* content seen so far
        # `tag['start'] - section_raw_start` is the tag's raw position relative to the section start.
        # `accumulated_tag_length` is how many characters we've skipped *before* this tag.
        effective_clean_tag_start = (
            tag["start"] - section_raw_start
        ) - accumulated_tag_length

        # If the target clean position is *after* where this tag would have started
        # in the cleaned text, it means this tag was removed before clean_pos,
        # so we need to add its length to our offset.
        if clean_pos > effective_clean_tag_start:
            accumulated_tag_length += tag["length"]
        else:
            # If clean_pos is before or at the effective start of the tag,
            # subsequent tags won't affect the mapping for this clean_pos.
            break

    # The raw position is the section's raw start, plus the position within
    # the clean text, plus the total length of all tags removed before that point.
    return section_raw_start + clean_pos + accumulated_tag_length


def create_directory(directory: str):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def cleanup_filename(name: str) -> str:
    """
    Cleans a string for use as a filename.
    (Identical logic to previous stages)
    """
    name = str(name)
    name = re.sub(r'[\\/:?*<>|"\']', "", name).strip()
    name = re.sub(r"[\s_]+", "_", name)
    return name[:50].strip("_")


def get_most_specific_heading(section_data: dict) -> str:
    """
    Retrieves the most specific heading text for a section.
    (Identical logic to previous stages)
    """
    current_level = section_data.get("level", 0)
    for level_num in range(current_level, 0, -1):
        level_key = f"level_{level_num}"
        heading_text = section_data.get(level_key)
        if heading_text:
            return heading_text
    return section_data.get("chapter_name", "Untitled_Section")


# --- Core Splitting Logic ---


def split_paragraph_by_sentences(paragraph: str, max_tokens: int) -> list[str]:
    """
    Splits a single paragraph string into smaller chunks based on sentence
    boundaries, ensuring no chunk exceeds `max_tokens`. If a single sentence
    exceeds `max_tokens`, it's split by words.

    Args:
        paragraph: The text of the paragraph (assumed cleaned).
        max_tokens: The maximum token limit for each resulting chunk.

    Returns:
        A list of strings, where each string is a chunk of the original
        paragraph, respecting sentence boundaries where possible.
    """
    # Simple sentence splitting based on common terminators followed by space/newline.
    # This might not be perfect for all edge cases (e.g., abbreviations).
    sentence_pattern = re.compile(
        r"(?<=[.!?])(?:\s+|\n+)"
    )  # Positive lookbehind for terminator
    sentences = sentence_pattern.split(paragraph)
    # Filter out empty strings that can result from splitting
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if not sentences:  # Handle empty or whitespace-only paragraphs
        return [paragraph] if paragraph.strip() else []

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # --- Handle sentences that are themselves too long ---
        if sentence_tokens > max_tokens:
            # If there's a pending chunk, save it first
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0

            # Split the oversized sentence by words
            words = sentence.split()
            temp_word_chunk = []
            temp_word_tokens = 0
            for word in words:
                # Estimate token count for word + space
                word_token_estimate = count_tokens(word + " ")
                # If adding the word exceeds limit, finalize the temp word chunk
                if (
                    temp_word_tokens > 0
                    and temp_word_tokens + word_token_estimate > max_tokens
                ):
                    chunks.append(" ".join(temp_word_chunk))
                    temp_word_chunk = [word]
                    temp_word_tokens = word_token_estimate
                else:
                    temp_word_chunk.append(word)
                    temp_word_tokens += word_token_estimate
            # Add any remaining words in the temp chunk
            if temp_word_chunk:
                chunks.append(" ".join(temp_word_chunk))

            # Reset for the next sentence (current oversized one is fully processed)
            current_chunk_sentences = []
            current_tokens = 0
            continue  # Move to the next sentence

        # --- Handle normal sentences ---
        # If adding the current sentence would exceed max_tokens, finalize the current chunk
        if current_tokens > 0 and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk_sentences))
            # Start a new chunk with the current sentence
            current_chunk_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            # Add the sentence to the current chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Add any remaining sentences in the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    # Filter out any potentially empty chunks created during processing
    return [c for c in chunks if c]


def split_large_section_content(cleaned_content: str, max_tokens: int) -> list[dict]:
    """
    Splits the cleaned content of a large section into smaller sub-chunks.

    Prioritizes splitting by paragraph breaks (`\\n\\s*\\n+`). If a paragraph
    itself exceeds `max_tokens`, it's further split by sentences using
    `split_paragraph_by_sentences`.

    Args:
        cleaned_content: The cleaned text content of the section to be split.
        max_tokens: The maximum token limit for any resulting sub-chunk.

    Returns:
        A list of dictionaries, where each dictionary represents a sub-chunk
        and contains:
        - 'content': The text content of the sub-chunk.
        - 'clean_start_pos': Start position relative to the input `cleaned_content`.
        - 'clean_end_pos': End position relative to the input `cleaned_content`.
        - 'token_count': Token count of the sub-chunk's content.
    """
    # Split content by double (or more) newlines to get paragraphs
    para_pattern = re.compile(r"\n\s*\n+")  # Matches one or more blank lines
    paragraphs = para_pattern.split(cleaned_content)

    split_sub_chunks = (
        []
    )  # Stores the final list of {'content', 'clean_start_pos', ...} dicts
    current_clean_pos = 0  # Track position within the *cleaned_content*

    for para_text in paragraphs:
        para_text_stripped = para_text.strip()
        if not para_text_stripped:
            # Advance position past the paragraph break itself
            match = para_pattern.search(cleaned_content, current_clean_pos)
            if match:
                current_clean_pos = match.end()
            continue  # Skip empty paragraphs

        para_tokens = count_tokens(para_text_stripped)
        para_clean_start_pos = cleaned_content.find(para_text, current_clean_pos)
        if para_clean_start_pos == -1:  # Should not happen if logic is correct
            print(
                f"WARN: Could not find paragraph start pos. Content:\n{para_text[:100]}..."
            )
            para_clean_start_pos = current_clean_pos  # Best guess

        # --- Handle paragraphs exceeding max_tokens ---
        if para_tokens > max_tokens:
            # Split the oversized paragraph by sentences
            sentence_chunks = split_paragraph_by_sentences(
                para_text_stripped, max_tokens
            )
            current_sentence_offset = 0
            for sentence_chunk in sentence_chunks:
                sentence_tokens = count_tokens(sentence_chunk)
                # Calculate start/end relative to the *original cleaned_content*
                chunk_clean_start = para_clean_start_pos + current_sentence_offset
                chunk_clean_end = chunk_clean_start + len(sentence_chunk)
                split_sub_chunks.append(
                    {
                        "content": sentence_chunk,
                        "clean_start_pos": chunk_clean_start,
                        "clean_end_pos": chunk_clean_end,
                        "token_count": sentence_tokens,
                    }
                )
                # Update offset for the next sentence chunk within this paragraph
                # Need to find the actual start of the next part in the original para_text
                # This is tricky if sentences were split/rejoined. Approximate by length.
                current_sentence_offset += len(sentence_chunk)
                # Add 1 or more for the space/newline delimiter that was likely removed
                # This isn't perfect but helps keep positions somewhat aligned.
                while (
                    current_sentence_offset < len(para_text)
                    and para_text[current_sentence_offset].isspace()
                ):
                    current_sentence_offset += 1

        # --- Handle paragraphs within the token limit ---
        else:
            chunk_clean_start = para_clean_start_pos
            chunk_clean_end = chunk_clean_start + len(
                para_text
            )  # Use original length before strip
            split_sub_chunks.append(
                {
                    "content": para_text_stripped,  # Store stripped version
                    "clean_start_pos": chunk_clean_start,
                    "clean_end_pos": chunk_clean_end,
                    "token_count": para_tokens,
                }
            )

        # Update current_clean_pos for the next paragraph search
        current_clean_pos = para_clean_start_pos + len(para_text)
        # Advance past potential paragraph break characters
        match = para_pattern.search(cleaned_content, current_clean_pos)
        if match and match.start() == current_clean_pos:
            current_clean_pos = match.end()

    # --- Combine adjacent small chunks resulting from splitting ---
    # This logic might be better placed within the paragraph/sentence splitting,
    # but doing it here ensures chunks are reasonably sized after initial splits.
    # Re-using the merge logic from Stage 2 might be complex due to position tracking.
    # For now, we return the potentially numerous small chunks from sentence splitting.
    # A simpler combination pass could be added if needed.

    return split_sub_chunks


# --- Helper Functions for Processing and Saving ---


def _generate_output_filename(input_filename: str, part_index: int) -> str:
    """
    Generates the output filename for a final chunk based on the input
    section filename and the part index (0 for unsplit, 1+ for split parts).

    Example: 001_005_Some_Heading.json -> 001_005_Some_Heading_001.json
    """
    # Try to parse the input filename (e.g., "001_005_Some_Heading.json")
    match = re.match(r"(\d+)_(\d+)_(.+)\.json", input_filename)
    if match:
        chap_num_str, sec_idx_str, base_name = match.groups()
        # Remove potential existing "_PartX" or "_00X" suffix from base_name if script was rerun
        base_name = re.sub(r"(_part|_Part|_)\d+$", "", base_name)
        # Ensure the base name is filesystem-safe
        clean_base_name = cleanup_filename(base_name)
        # Format: ChapterNum_SectionIndex_CleanBaseName_PartIndex.json
        return f"{chap_num_str}_{sec_idx_str}_{clean_base_name}_{part_index:03d}.json"
    else:
        # Fallback if the input filename format is unexpected
        base = os.path.splitext(input_filename)[0]
        base = re.sub(r"(_part|_Part|_)\d+$", "", base)  # Clean potential old suffix
        clean_base = cleanup_filename(base)
        return f"{clean_base}_{part_index:03d}.json"


def _handle_section_within_limit(
    section_data: dict, input_filename: str, output_dir: str
) -> int:
    """
    Saves a section that is within the token limit as a single chunk (Part 0).

    Args:
        section_data: The data dictionary for the section.
        input_filename: The original filename of the section JSON.
        output_dir: The directory to save the final chunk JSON.

    Returns:
        1 if saved successfully, 0 otherwise.
    """
    original_token_count = section_data.get(
        "token_count", 0
    )  # Get original count before potential pop
    print(
        f"  Section token count ({original_token_count}) is within limit. Saving as single chunk."
    )

    # Prepare final chunk data
    final_chunk_data = section_data.copy()
    final_chunk_data["chunk_token_count"] = (
        original_token_count  # Add specific chunk token count
    )
    final_chunk_data["chunk_part_number"] = 0  # Indicate it's the only part
    final_chunk_data.pop(
        "token_count", None
    )  # Remove the original section token count key

    # Generate output filename (Part 0)
    output_filename = _generate_output_filename(input_filename, part_index=0)
    output_filepath = os.path.join(output_dir, output_filename)

    # Save the chunk
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(final_chunk_data, f, indent=2, ensure_ascii=False)
        return 1  # Saved 1 chunk
    except Exception as e:
        print(f"  ERROR saving chunk {output_filename}: {e}")
        return 0


def _handle_section_needs_splitting(
    section_data: dict, input_filename: str, original_md_dir: str, output_dir: str
) -> int:
    """
    Handles splitting a section that exceeds MAX_TOKENS. Splits content, maps
    positions, deduplicates, updates metadata, and saves unique sub-chunks.

    Args:
        section_data: Data dictionary of the large section.
        input_filename: Original filename of the section JSON.
        original_md_dir: Directory containing original markdown files.
        output_dir: Directory to save the final chunk JSON files.

    Returns:
        The number of unique chunks saved after splitting.
    """
    original_token_count = section_data.get(
        "token_count", 0
    )  # Original count of the whole section
    cleaned_content = section_data.get(
        "content", ""
    )  # Cleaned content of the whole section
    section_raw_start = section_data.get("start_pos")  # Start pos in original raw MD
    section_raw_end = section_data.get("end_pos")  # End pos in original raw MD
    source_md_filename = section_data.get("source_markdown_file")
    saved_chunk_count = 0

    print(
        f"  Section token count ({original_token_count}) exceeds limit ({MAX_TOKENS}). Splitting..."
    )

    # --- Pre-requisite checks ---
    if section_raw_start is None or section_raw_end is None or not source_md_filename:
        print(
            f"  ERROR: Missing 'start_pos', 'end_pos', or 'source_markdown_file' needed for splitting. Skipping {input_filename}."
        )
        return 0

    # --- Find and Read Original Markdown ---
    original_md_path = find_original_md(
        source_md_filename, original_md_dir
    )  # Use helper
    if not original_md_path:
        print(
            f"  ERROR: Could not locate original MD file '{source_md_filename}'. Skipping split."
        )
        return 0
    try:
        with open(original_md_path, "r", encoding="utf-8") as f:
            raw_chapter_content = f.read()
    except Exception as e:
        print(
            f"  ERROR: Could not read original MD file '{original_md_path}': {e}. Skipping split."
        )
        return 0

    # --- Extract Tags and Split Content ---
    # Get all tags from the raw chapter content
    full_tag_mapping = extract_tag_mapping(raw_chapter_content)
    # Filter to get only tags within this section's raw boundaries
    section_tag_mapping = [
        tag
        for tag in full_tag_mapping
        if tag["start"] >= section_raw_start and tag["end"] <= section_raw_end
    ]

    # Split the cleaned content into sub-chunks based on paragraphs/sentences
    split_sub_chunks = split_large_section_content(cleaned_content, MAX_TOKENS)
    print(f"  Split into {len(split_sub_chunks)} potential sub-chunks.")

    # --- Process and Save Unique Sub-Chunks ---
    seen_content_hashes = set()  # Track hashes to deduplicate *within this section*
    split_part_index = 0  # Index for the output parts (starts at 1)

    for sub_chunk in split_sub_chunks:
        # Calculate content hash for deduplication
        content_hash = hashlib.md5(sub_chunk["content"].encode("utf-8")).hexdigest()
        if content_hash in seen_content_hashes:
            print(
                f"    Skipping duplicate content sub-chunk (Hash: {content_hash[:8]}...)."
            )
            continue
        seen_content_hashes.add(content_hash)

        split_part_index += 1  # Increment part number for unique chunks

        # Map the sub-chunk's clean start/end positions back to raw positions
        chunk_raw_start = map_clean_to_raw_pos(
            sub_chunk["clean_start_pos"], section_raw_start, section_tag_mapping
        )
        chunk_raw_end = map_clean_to_raw_pos(
            sub_chunk["clean_end_pos"], section_raw_start, section_tag_mapping
        )

        # Prepare the data dictionary for the final chunk JSON
        final_chunk_data = section_data.copy()  # Start with original section metadata
        final_chunk_data["content"] = sub_chunk["content"]  # Use the sub-chunk content
        final_chunk_data["chunk_token_count"] = sub_chunk[
            "token_count"
        ]  # Sub-chunk token count
        final_chunk_data["chunk_part_number"] = (
            split_part_index  # Part number (1, 2, ...)
        )
        final_chunk_data["start_pos"] = chunk_raw_start  # Updated raw start position
        final_chunk_data["end_pos"] = chunk_raw_end  # Updated raw end position
        final_chunk_data.pop(
            "token_count", None
        )  # Remove original section's token count

        # Append "(Part X)" to the most specific heading for identification
        heading_key = f"level_{final_chunk_data.get('level', 0)}"
        if heading_key in final_chunk_data:
            heading_text = final_chunk_data[heading_key]
            # Avoid adding suffix multiple times if script is rerun
            if not re.search(r" \(Part \d+\)$", heading_text):
                final_chunk_data[heading_key] = (
                    f"{heading_text} (Part {split_part_index})"
                )
        # Consider adding to chapter_name as well if no specific heading?

        # Generate output filename and save the chunk
        output_filename = _generate_output_filename(input_filename, split_part_index)
        output_filepath = os.path.join(output_dir, output_filename)
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(final_chunk_data, f, indent=2, ensure_ascii=False)
            saved_chunk_count += 1
        except Exception as e:
            print(f"  ERROR saving chunk {output_filename}: {e}")
            # Continue processing other chunks even if one fails to save

    print(f"  Saved {saved_chunk_count} unique chunks after splitting.")
    return saved_chunk_count


def process_paged_section_json(
    section_json_path: str, original_md_dir: str, output_dir: str
) -> int:
    """
    Processes a single paged section JSON file from Stage 3.

    Loads the section data, checks if it needs splitting based on `MAX_TOKENS`.
    Calls the appropriate handler (`_handle_section_within_limit` or
    `_handle_section_needs_splitting`) to process and save the final chunk(s).

    Args:
        section_json_path: Path to the input paged section JSON file.
        original_md_dir: Directory containing original markdown files.
        output_dir: Directory to save the final chunk JSON files.

    Returns:
        The number of final chunks saved for this section (0 or 1 if not split,
        potentially more if split, 0 on error).
    """
    input_filename = os.path.basename(section_json_path)
    try:
        print(f"Processing Stage 4 for: {input_filename}")

        # Load section data
        with open(section_json_path, "r", encoding="utf-8") as f:
            section_data = json.load(f)

        # Check token count against the limit
        original_token_count = section_data.get("token_count", 0)

        if original_token_count <= MAX_TOKENS:
            # Section is within limit, save as single chunk
            return _handle_section_within_limit(
                section_data, input_filename, output_dir
            )
        else:
            # Section exceeds limit, needs splitting
            return _handle_section_needs_splitting(
                section_data, input_filename, original_md_dir, output_dir
            )

    except Exception as e:
        print(f"\nERROR processing section file {input_filename} in Stage 4: {e}")
        print(traceback.format_exc())
        return 0  # Indicate failure for this section file


def main():
    """
    Main execution function for Stage 4.

    Finds paged section JSON files, sorts them, processes each using
    `process_paged_section_json` (which handles potential splitting and
    deduplication), and saves the final chunk JSON files.
    """
    print("-" * 50)
    print("Running Stage 4: Split Large Sections, Deduplicate, and Finalize Chunks")
    print(f"Input paged section directory: {INPUT_DIR}")
    print(f"Original MD directory        : {ORIGINAL_MD_DIR}")
    print(f"Output final chunk directory : {OUTPUT_DIR}")
    print("-" * 50)

    # Ensure output directory exists
    create_directory(OUTPUT_DIR)

    # --- Find and Sort Input Files ---
    paged_section_files = []
    try:
        all_files = os.listdir(INPUT_DIR)
        paged_section_files = [
            os.path.join(INPUT_DIR, f) for f in all_files if f.endswith(".json")
        ]
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return
    except OSError as e:
        print(f"ERROR: Could not list files in input directory '{INPUT_DIR}': {e}")
        return

    if not paged_section_files:
        print(f"No paged section JSON files found in {INPUT_DIR}. Exiting.")
        return

    # Sort files (natural sort preferred)
    if natsort:
        paged_section_files = natsort.natsorted(paged_section_files)
        print(
            f"Found and naturally sorted {len(paged_section_files)} paged section JSON files."
        )
    else:
        paged_section_files.sort()
        print(
            f"Found {len(paged_section_files)} paged section JSON files (standard sort)."
        )
        if natsort is None:
            print(
                "INFO: Install 'natsort' for potentially better file ordering (pip install natsort)."
            )

    # --- Process Files ---
    total_final_chunks_saved = 0
    processed_section_files_count = 0
    failed_section_files_count = 0

    # Setup progress bar if tqdm is available
    file_iterator = paged_section_files
    if tqdm:
        file_iterator = tqdm(
            paged_section_files,
            desc="Stage 4 Processing Sections",
            unit="section",
            ncols=100,
        )

    # Process each paged section file
    for section_file_path in file_iterator:
        # This function returns the number of *final chunks* saved for the section
        chunks_saved_for_section = process_paged_section_json(
            section_file_path, ORIGINAL_MD_DIR, OUTPUT_DIR
        )
        if chunks_saved_for_section > 0:
            total_final_chunks_saved += chunks_saved_for_section
            processed_section_files_count += 1
        else:
            # If 0 chunks were saved, consider the processing of this section file failed
            failed_section_files_count += 1

    # --- Print Summary ---
    print("-" * 50)
    print("Stage 4 Summary:")
    print(
        f"Paged section files processed successfully: {processed_section_files_count}"
    )
    print(f"Paged section files failed processing   : {failed_section_files_count}")
    print(f"Total final chunks saved                : {total_final_chunks_saved}")
    print(f"Output final chunk JSON files are in    : {OUTPUT_DIR}")
    print("-" * 50)


if __name__ == "__main__":
    main()
