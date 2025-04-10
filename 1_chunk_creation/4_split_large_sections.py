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


# --- Helper Functions for Processing and Saving ---

def find_original_md(filename: str, md_dir: str) -> str | None:
    """
    Constructs the full path to an original markdown file and checks existence.

    Args:
        filename: The base name of the markdown file (e.g., "Chapter_1.md").
        md_dir: The directory where the original markdown files are stored.

    Returns:
        The full path if the file exists, otherwise None.
    """
    if not filename or not md_dir:
        return None
    full_path = os.path.join(md_dir, filename)
    if os.path.isfile(full_path):
        return full_path
    else:
        # Attempt fallback: Check if filename already contains the directory prefix
        # (e.g., if source_markdown_file was accidentally saved with the path)
        if os.path.isfile(filename):
             print(f"  WARN: Found original MD file using the provided name '{filename}' directly, which might indicate an unexpected format for 'source_markdown_file'.")
             return filename
        return None


# --- Core Splitting Logic ---

def split_paragraph_by_sentences(paragraph, max_tokens=MAX_TOKENS):
    """Split a large paragraph (cleaned text) into smaller chunks of roughly even size."""
    # Pattern to match sentences by whitespace or newline, uses positive lookahead
    # to avoid dropping sentence terminators
    # sentence_pattern = re.compile(r'[^.!?]+[.!?]') # Original attempt - might miss things
    sentence_pattern = re.compile(r'(?<=[.!?])(?:\s+|\n+)') # Lookbehind for terminator

    # Split text into potential sentences
    sentences = sentence_pattern.split(paragraph)
    # Filter out any empty strings that might result from the split
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if not sentences:
        # If splitting results in nothing, return the original paragraph as one chunk
        # This might happen if the paragraph has no standard sentence terminators
        return [paragraph] if paragraph.strip() else []

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # --- Handle oversized sentences ---
        if sentence_tokens > max_tokens:
            # 1. Finalize the chunk accumulated *before* this oversized sentence
            if current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0

            # 2. Split the long sentence itself by words (arbitrary split)
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            for word in words:
                # Estimate token count for word + space
                word_tokens = count_tokens(word + ' ')

                if temp_tokens > 0 and temp_tokens + word_tokens > max_tokens:
                    # This word pushes over the limit, finalize previous temp chunk
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_tokens = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens

            # Add any remaining words in the temp chunk
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))

            # Reset current chunk tracking after handling the oversized sentence
            current_chunk_sentences = []
            current_tokens = 0
            continue # Move to the next sentence

        # --- Handle normal sentences ---
        # If adding this sentence would exceed max tokens and we already have content
        elif current_tokens > 0 and current_tokens + sentence_tokens > max_tokens:
            # Finalize current chunk
            chunks.append(' '.join(current_chunk_sentences))
            # Start a new chunk with this sentence
            current_chunk_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            # Add sentence to the current chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Add the final accumulated chunk if any sentences are left
    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))

    # Ensure no empty chunks are returned
    return [c for c in chunks if c]

def split_large_section_content(cleaned_content, max_tokens=MAX_TOKENS):
    """
    Splits the cleaned content of a large section into smaller sub-chunks using
    the greedy paragraph and sentence combination strategy.
    """
    # Find paragraph boundaries in the cleaned content
    para_pattern = re.compile(r'\n\s*\n+') # Use original script's pattern
    para_matches = list(para_pattern.finditer(cleaned_content))

    # Get end positions of paragraph separators (start of next paragraph)
    para_boundaries = [m.end() for m in para_matches]

    # Define paragraph start/end positions
    split_sub_chunks = []

    current_chunk_paras_clean_text = []
    current_chunk_clean_start_pos = 0 # Relative to cleaned_content start
    current_chunk_tokens = 0

    # Iterate through paragraphs defined by boundaries
    for i in range(len(para_boundaries) + 1):
        # Determine the start and end of the current paragraph slice
        para_slice_start = para_boundaries[i-1] if i > 0 else 0
        para_slice_end = para_boundaries[i] if i < len(para_boundaries) else len(cleaned_content)

        # Extract the paragraph text itself (including internal whitespace)
        para_text_raw = cleaned_content[para_slice_start:para_slice_end]
        para_text_stripped = para_text_raw.strip() # For token counting and content storage

        if not para_text_stripped:  # Skip effectively empty paragraphs
            continue

        para_tokens = count_tokens(para_text_stripped)

        # Find the actual start position of the stripped text within the raw slice
        # This is needed for accurate clean_start_pos calculation
        try:
            strip_offset = para_text_raw.index(para_text_stripped)
            para_clean_start_in_section = para_slice_start + strip_offset
        except ValueError:
             # Should not happen if para_text_stripped is not empty
             para_clean_start_in_section = para_slice_start


        # --- Handle oversized paragraphs by splitting them first ---
        if para_tokens > max_tokens:
            # 1. Finalize the chunk accumulated *before* this oversized paragraph
            if current_chunk_paras_clean_text:
                chunk_clean_text = '\n\n'.join(current_chunk_paras_clean_text)
                # End position is where the oversized paragraph's raw slice started
                chunk_clean_end_pos = para_slice_start
                split_sub_chunks.append({
                    'content': chunk_clean_text,
                    'clean_start_pos': current_chunk_clean_start_pos,
                    'clean_end_pos': chunk_clean_end_pos,
                    'token_count': current_chunk_tokens
                })
                # Reset tracking for the chunks coming from the split paragraph
                current_chunk_paras_clean_text = []
                current_chunk_tokens = 0

            # 2. Split the oversized paragraph itself by sentences
            sentence_chunks_text = split_paragraph_by_sentences(para_text_stripped, max_tokens)
            current_sentence_offset_in_para = 0 # Track position within the stripped paragraph

            for sentence_chunk_text in sentence_chunks_text:
                sentence_chunk_tokens = count_tokens(sentence_chunk_text)
                # Calculate start/end relative to the *section*'s cleaned_content start
                # Start is the paragraph's clean start + offset within the stripped para
                chunk_clean_start = para_clean_start_in_section + current_sentence_offset_in_para
                chunk_clean_end = chunk_clean_start + len(sentence_chunk_text) # Approx end

                split_sub_chunks.append({
                    'content': sentence_chunk_text,
                    'clean_start_pos': chunk_clean_start,
                    'clean_end_pos': chunk_clean_end, # Note: This end pos might be slightly off due to sentence joining spaces
                    'token_count': sentence_chunk_tokens
                })

                # Update start offset for next sentence chunk within the paragraph
                # Find where the next sentence actually starts in the stripped paragraph text
                try:
                    # Search *after* the current sentence chunk's text
                    next_start_in_para = para_text_stripped.index(
                        sentence_chunk_text, current_sentence_offset_in_para
                        ) + len(sentence_chunk_text)
                    # Skip potential whitespace between sentences
                    while next_start_in_para < len(para_text_stripped) and para_text_stripped[next_start_in_para].isspace():
                        next_start_in_para += 1
                    current_sentence_offset_in_para = next_start_in_para
                except ValueError:
                    # If not found (e.g., last sentence), break or set to end
                    current_sentence_offset_in_para = len(para_text_stripped)


            # After splitting the oversized paragraph, the next chunk will start
            # at the beginning of the *next* paragraph's slice.
            # Set the start position for a potential *new* chunk starting after this split one.
            current_chunk_clean_start_pos = para_slice_end # Start next chunk after the para break
            current_chunk_paras_clean_text = [] # Ensure reset
            current_chunk_tokens = 0 # Ensure reset
            continue  # Move to the next paragraph in the outer loop

        # --- Handle normal paragraphs (not oversized themselves) ---
        # Check if adding this paragraph exceeds max tokens
        if current_chunk_tokens > 0 and current_chunk_tokens + para_tokens > max_tokens:
            # Finalize the current chunk *before* adding this paragraph
            chunk_clean_text = '\n\n'.join(current_chunk_paras_clean_text)
            # End position is where this paragraph's raw slice started
            chunk_clean_end_pos = para_slice_start

            split_sub_chunks.append({
                'content': chunk_clean_text,
                'clean_start_pos': current_chunk_clean_start_pos,
                'clean_end_pos': chunk_clean_end_pos,
                'token_count': current_chunk_tokens
            })

            # Start a new chunk with the current paragraph
            current_chunk_paras_clean_text = [para_text_stripped]
            current_chunk_tokens = para_tokens
            current_chunk_clean_start_pos = para_clean_start_in_section # Use actual text start
        else:
            # Add paragraph to the current chunk
            # If this is the first paragraph of a new chunk, set its start position
            if not current_chunk_paras_clean_text:
                current_chunk_clean_start_pos = para_clean_start_in_section
            current_chunk_paras_clean_text.append(para_text_stripped)
            current_chunk_tokens += para_tokens

    # Add the final accumulated chunk after the loop finishes
    if current_chunk_paras_clean_text:
        chunk_clean_text = '\n\n'.join(current_chunk_paras_clean_text)
        # End position is the end of the original cleaned content
        chunk_clean_end_pos = len(cleaned_content)

        split_sub_chunks.append({
            'content': chunk_clean_text,
            'clean_start_pos': current_chunk_clean_start_pos,
            'clean_end_pos': chunk_clean_end_pos,
            'token_count': current_chunk_tokens
        })

    return split_sub_chunks


# --- Helper Functions for Processing and Saving ---
# Note: _generate_output_filename and _handle_section* functions are removed
# as they are replaced by the logic within the new process_paged_section_json


def process_paged_section_json(section_json_path, original_md_dir, output_dir):
    """Processes a single paged section JSON using the user-provided logic."""
    saved_chunk_count = 0
    input_filename = os.path.basename(section_json_path) # Moved up for logging

    try:
        print(f"Processing Stage 4 for: {input_filename}")

        # 1. Read section JSON from Stage 3
        with open(section_json_path, 'r', encoding='utf-8') as f:
            section_data = json.load(f)

        original_token_count = section_data.get('token_count', 0)
        # Use 'content' key as per original script and user code context
        cleaned_content = section_data.get('content', '')
        original_raw_start = section_data.get('start_pos')
        original_raw_end = section_data.get('end_pos') # Needed for tag filtering
        source_md_filename = section_data.get('source_markdown_file')

        # --- Check if splitting is needed ---
        if original_token_count <= MAX_TOKENS:
            # No splitting needed, just format and save
            print(f"  Section token count ({original_token_count}) is within limit. Saving as single chunk.")
            final_chunk_data = section_data.copy()
            final_chunk_data['chunk_part_number'] = 0  # Not split
            # Add chunk_token_count for consistency, even if same as original
            final_chunk_data['chunk_token_count'] = original_token_count
            final_chunk_data.pop('token_count', None) # Remove original key

            # Find a clean filename: chX_secY_name_part000.json
            match = re.match(r'(\d+)_(\d+)_(.+)\.json', input_filename)
            if match:
                chap_num_str, sec_idx_str, base_name = match.groups()
                # Clean base name similar to original script's helper
                clean_base_name = cleanup_filename(base_name)
                # Use user's preferred format _partXXX.json
                output_filename = f"{chap_num_str}_{sec_idx_str}_{clean_base_name}_part000.json"
            else:
                # Fallback filename if pattern doesn't match
                base = os.path.splitext(input_filename)[0]
                clean_base = cleanup_filename(base)
                output_filename = f"{clean_base}_part000.json"

            output_filepath = os.path.join(output_dir, output_filename)

            # Save the single chunk
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(final_chunk_data, f, indent=2, ensure_ascii=False)
                saved_chunk_count = 1
            except Exception as e:
                 print(f"  ERROR saving chunk {output_filename}: {e}")
                 saved_chunk_count = 0 # Explicitly set to 0 on save error

        else:
            # --- Splitting is needed ---
            print(f"  Section token count ({original_token_count}) exceeds limit ({MAX_TOKENS}). Splitting...")

            # Use original script's checks for necessary data
            if original_raw_start is None or original_raw_end is None or not source_md_filename:
                print(f"  ERROR: Missing 'start_pos', 'end_pos', or 'source_markdown_file' needed for splitting. Skipping {input_filename}.")
                return 0 # Return 0 chunks saved

            # Read original MD file for raw content and tag mapping
            # Use find_original_md helper from original script for robustness
            original_md_path = find_original_md(source_md_filename, original_md_dir)
            if not original_md_path:
                 print(f"  ERROR: Could not locate original MD file '{source_md_filename}'. Skipping split.")
                 return 0

            try:
                with open(original_md_path, 'r', encoding='utf-8') as f:
                    raw_chapter_content = f.read()
            except Exception as e:
                print(f"  ERROR: Could not read original MD file '{original_md_path}': {e}. Skipping split.")
                return 0

            # Extract tag mapping relevant to this section's raw slice
            full_tag_mapping = extract_tag_mapping(raw_chapter_content)
            section_tag_mapping = [
                tag for tag in full_tag_mapping
                if tag['start'] >= original_raw_start and tag['end'] <= original_raw_end
            ]

            # Split the cleaned content using the new function
            split_sub_chunks = split_large_section_content(cleaned_content, MAX_TOKENS)
            print(f"  Split into {len(split_sub_chunks)} potential sub-chunks.")

            # Deduplicate and save
            seen_content_hashes = set()
            split_part_index = 0  # Start part numbering from 1

            for sub_chunk in split_sub_chunks:
                # Calculate content hash for deduplication
                content_hash = hashlib.md5(sub_chunk['content'].encode('utf-8')).hexdigest()

                if content_hash in seen_content_hashes:
                    print(f"    Skipping duplicate content chunk (hash: {content_hash[:8]}...)")
                    continue

                seen_content_hashes.add(content_hash)
                split_part_index += 1  # Increment part number for unique chunks

                # Map clean positions back to raw positions using helper from original script
                chunk_raw_start = map_clean_to_raw_pos(sub_chunk['clean_start_pos'], original_raw_start, section_tag_mapping)
                chunk_raw_end = map_clean_to_raw_pos(sub_chunk['clean_end_pos'], original_raw_start, section_tag_mapping)

                # Create the final chunk data
                final_chunk_data = section_data.copy()  # Start with original section metadata
                final_chunk_data['content'] = sub_chunk['content']
                final_chunk_data['chunk_token_count'] = sub_chunk['token_count']
                final_chunk_data['chunk_part_number'] = split_part_index
                final_chunk_data['start_pos'] = chunk_raw_start  # Update raw positions
                final_chunk_data['end_pos'] = chunk_raw_end

                # Add (Part X) to the most specific heading using helper from original script
                heading_key = f"level_{final_chunk_data.get('level', 0)}"
                if heading_key in final_chunk_data:
                     heading_text = final_chunk_data[heading_key]
                     # Avoid adding suffix multiple times if script is rerun
                     if not re.search(r" \(Part \d+\)$", heading_text):
                         final_chunk_data[heading_key] = f"{heading_text} (Part {split_part_index})"

                # Remove original token_count
                final_chunk_data.pop('token_count', None)

                # Construct filename using user's preferred format _partXXX.json
                match = re.match(r'(\d+)_(\d+)_(.+)\.json', input_filename)
                if match:
                    chap_num_str, sec_idx_str, base_name = match.groups()
                    # Clean base name
                    clean_base_name = cleanup_filename(base_name)
                    # Remove potential existing suffix before adding new one
                    clean_base_name = re.sub(r"(_part|_Part|_)\d+$", "", clean_base_name)
                    output_filename = f"{chap_num_str}_{sec_idx_str}_{clean_base_name}_part{split_part_index:03d}.json"
                else:
                    # Fallback filename if pattern doesn't match
                    base = os.path.splitext(input_filename)[0]
                    clean_base = cleanup_filename(base)
                    clean_base = re.sub(r"(_part|_Part|_)\d+$", "", clean_base) # Clean suffix here too
                    output_filename = f"{clean_base}_part{split_part_index:03d}.json"

                output_filepath = os.path.join(output_dir, output_filename)

                # Save the chunk
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(final_chunk_data, f, indent=2, ensure_ascii=False)
                    saved_chunk_count += 1
                except Exception as e:
                    print(f"  ERROR saving chunk {output_filename}: {e}")
                    # Continue processing other chunks

            print(f"  Saved {saved_chunk_count} unique chunks after splitting.")

    except Exception as e:
        print(f"\nERROR processing section file {input_filename} in Stage 4: {e}")
        print(traceback.format_exc())
        return 0 # Indicate failure for this section file

    return saved_chunk_count


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
