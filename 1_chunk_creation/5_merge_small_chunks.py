"""
Stage 5: Final Merge of Ultra-Small Chunks.

Purpose:
Performs a final cleanup pass on the chunk JSON files generated by Stage 4.
It specifically targets "ultra-small" chunks (those below `ULTRA_SMALL_THRESHOLD`
tokens, often resulting from splitting headings or very short paragraphs).
It attempts to merge these small chunks into adjacent chunks (either preceding
or succeeding) based on preferred directions (headings merge forward, content
merges backward) and fallback directions, while respecting the `MAX_TOKENS`
limit and ensuring chunks belong to the same original chapter. Chunks that
cannot be merged are kept as they are. The final set of chunks is saved to a
new output directory.

Input: Final chunk JSON files from `INPUT_DIR` (Stage 4 output).
Output: Final merged chunk JSON files in `OUTPUT_DIR`.
"""

import os
import json
import traceback  # Retained for detailed error logging

try:
    import natsort
except ImportError:
    natsort = None  # Optional dependency for natural sorting.

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("WARN: 'tiktoken' not installed. Token counts will be estimates.")

# --- Configuration & Constants ---
DEFAULT_INPUT_DIR = "2D_final_chunks"  # Directory containing Stage 4 chunk JSON files.
DEFAULT_OUTPUT_DIR = (
    "2E_final_merged_chunks"  # Directory to save the final merged chunk JSON files.
)

# Threshold below which chunks are considered "ultra-small" and targeted for merging.
ULTRA_SMALL_THRESHOLD = 25
# Maximum token limit for a chunk after merging.
MAX_TOKENS = 750

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
            # print(f"WARN: tiktoken encode failed ('{str(e)[:50]}...'). Falling back to estimate.") # Keep commented out
            return len(text) // 4
    else:
        # Estimate tokens if tokenizer isn't available
        return len(text) // 4


def create_directory(directory: str):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


# --- Core Logic ---


def load_and_sort_chunks(input_dir: str) -> list[dict] | None:
    """
    Loads all JSON files from the input directory, extracts their data,
    and sorts them using natural sorting based on filename.

    Args:
        input_dir: The directory containing the chunk JSON files.

    Returns:
        A list of dictionaries, where each dictionary contains the loaded data
        from a JSON file plus an added '_filename' key. Returns None if the
        input directory is not found or inaccessible. Returns empty list if
        directory exists but contains no JSON files.
    """
    all_chunks_data = []
    print(f"Loading chunks from: {input_dir}")
    try:
        # List and filter JSON files
        filenames = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        if not filenames:
            print(f"WARN: No JSON files found in {input_dir}")
            return []  # Return empty list, not None

        # Sort filenames using natural sort if available
        if natsort:
            filenames = natsort.natsorted(filenames)
            print(f"Found and naturally sorted {len(filenames)} chunk files.")
        else:
            filenames.sort()  # Standard sort as fallback
            print(f"Found {len(filenames)} chunk files (standard sort).")
            if natsort is None:
                print(
                    "INFO: Install 'natsort' for potentially better file ordering (pip install natsort)."
                )

        # Load data from each file
        files_read_error = 0
        for filename in filenames:
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_filename"] = filename  # Store original filename for saving later
                all_chunks_data.append(data)
            except (json.JSONDecodeError, OSError, Exception) as e:
                print(f"  ERROR reading or parsing {filename}: {e}. Skipping.")
                files_read_error += 1

        if files_read_error > 0:
            print(f"Warning: {files_read_error} files failed to load or parse.")
        print(f"Successfully loaded data for {len(all_chunks_data)} chunks.")
        return all_chunks_data

    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {input_dir}")
        return None  # Indicate critical error
    except OSError as e:
        print(f"ERROR: Could not access input directory '{input_dir}': {e}")
        return None  # Indicate critical error
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during chunk loading: {e}")
        traceback.print_exc()
        return None


def merge_final_small_chunks(
    chunks: list[dict], small_threshold: int, max_tokens: int
) -> list[dict]:
    """
    Performs the final merge pass for ultra-small chunks.

    Iterates through the sorted list of chunks. If a chunk is below
    `small_threshold`, it attempts to merge it with an adjacent chunk
    (preferring forward for headings, backward for content, with fallbacks)
    without exceeding `max_tokens`.

    Args:
        chunks: List of chunk dictionaries, assumed sorted correctly.
        small_threshold: Token count below which a chunk is considered for merging.
        max_tokens: Maximum allowed token count for a merged chunk.

    Returns:
        A new list containing the chunks after the final merge pass.
        Chunks that were merged into others are omitted.
    """
    if not chunks:
        return []

    # Create a new list for the result, don't modify the input list directly
    final_chunks_list = []
    # Keep track of which original chunks were merged away
    merged_flags = [False] * len(chunks)
    i = 0

    while i < len(chunks):
        # Skip chunks that have already been merged into a previous one
        if merged_flags[i]:
            i += 1
            continue

        current_chunk = chunks[i]
        current_chunk_tokens = current_chunk.get("chunk_token_count", 0)

        # If chunk is large enough, add it to the final list and move on
        if current_chunk_tokens >= small_threshold:
            final_chunks_list.append(current_chunk)
            i += 1
            continue

        # --- Chunk is smaller than threshold, attempt merging ---
        print(
            f"Found small chunk: {current_chunk.get('_filename')} ({current_chunk_tokens} tokens)"
        )
        # Simple check if content looks like just a heading
        is_heading_only = current_chunk.get("content", "").strip().startswith("#")
        merge_occurred = False

        # --- Preferred Merge Direction ---
        if is_heading_only:
            # Headings prefer merging FORWARD (into the next available chunk)
            next_available_idx = i + 1
            while next_available_idx < len(chunks) and merged_flags[next_available_idx]:
                next_available_idx += 1  # Find the next non-merged chunk

            if next_available_idx < len(chunks):
                next_chunk = chunks[next_available_idx]
                next_chunk_tokens = next_chunk.get("chunk_token_count", 0)
                # Check if same chapter and combined tokens <= max_tokens
                if (
                    current_chunk.get("chapter_number")
                    == next_chunk.get("chapter_number")
                    and current_chunk_tokens + next_chunk_tokens <= max_tokens
                ):

                    print(
                        f"  Merging forward (heading) into {next_chunk.get('_filename')}"
                    )
                    # Prepend current content to next chunk's content
                    next_chunk["content"] = (
                        f"{current_chunk.get('content', '')}\n\n{next_chunk.get('content', '')}"
                    )
                    # Recalculate token count for the merged chunk
                    next_chunk["chunk_token_count"] = count_tokens(
                        next_chunk["content"]
                    )
                    # Update start position to the current chunk's start
                    next_chunk["start_pos"] = current_chunk["start_pos"]
                    # Mark the current chunk as merged away
                    merged_flags[i] = True
                    merge_occurred = True
                    # Note: We don't add the merged chunk to final_chunks_list yet,
                    # it will be added when the loop reaches next_available_idx.
        else:
            # Non-headings (content) prefer merging BACKWARD (into the last chunk added to final_chunks_list)
            if final_chunks_list:  # Check if there's a previous chunk to merge into
                prev_chunk = final_chunks_list[-1]
                prev_chunk_tokens = prev_chunk.get("chunk_token_count", 0)
                # Check if same chapter and combined tokens <= max_tokens
                if (
                    current_chunk.get("chapter_number")
                    == prev_chunk.get("chapter_number")
                    and prev_chunk_tokens + current_chunk_tokens <= max_tokens
                ):

                    print(f"  Merging backward into {prev_chunk.get('_filename')}")
                    # Append current content to previous chunk's content
                    prev_chunk["content"] = (
                        f"{prev_chunk.get('content', '')}\n\n{current_chunk.get('content', '')}"
                    )
                    # Recalculate token count
                    prev_chunk["chunk_token_count"] = count_tokens(
                        prev_chunk["content"]
                    )
                    # Update end position to the current chunk's end
                    prev_chunk["end_pos"] = current_chunk["end_pos"]
                    # Mark current chunk as merged away
                    merged_flags[i] = True
                    merge_occurred = True

        # --- Fallback Merge Direction ---
        if not merge_occurred:
            if is_heading_only:
                # Fallback for heading: Merge BACKWARD
                if final_chunks_list:
                    prev_chunk = final_chunks_list[-1]
                    prev_chunk_tokens = prev_chunk.get("chunk_token_count", 0)
                    if (
                        current_chunk.get("chapter_number")
                        == prev_chunk.get("chapter_number")
                        and prev_chunk_tokens + current_chunk_tokens <= max_tokens
                    ):
                        print(
                            f"  Merging backward (heading fallback) into {prev_chunk.get('_filename')}"
                        )
                        prev_chunk["content"] = (
                            f"{prev_chunk.get('content', '')}\n\n{current_chunk.get('content', '')}"
                        )
                        prev_chunk["chunk_token_count"] = count_tokens(
                            prev_chunk["content"]
                        )
                        prev_chunk["end_pos"] = current_chunk["end_pos"]
                        merged_flags[i] = True
                        merge_occurred = True
            else:
                # Fallback for content: Merge FORWARD
                next_available_idx = i + 1
                while (
                    next_available_idx < len(chunks)
                    and merged_flags[next_available_idx]
                ):
                    next_available_idx += 1

                if next_available_idx < len(chunks):
                    next_chunk = chunks[next_available_idx]
                    next_chunk_tokens = next_chunk.get("chunk_token_count", 0)
                    if (
                        current_chunk.get("chapter_number")
                        == next_chunk.get("chapter_number")
                        and current_chunk_tokens + next_chunk_tokens <= max_tokens
                    ):
                        print(
                            f"  Merging forward (non-heading fallback) into {next_chunk.get('_filename')}"
                        )
                        next_chunk["content"] = (
                            f"{current_chunk.get('content', '')}\n\n{next_chunk.get('content', '')}"
                        )
                        next_chunk["chunk_token_count"] = count_tokens(
                            next_chunk["content"]
                        )
                        next_chunk["start_pos"] = current_chunk["start_pos"]
                        merged_flags[i] = True
                        merge_occurred = True

        # If still not merged, keep the small chunk as is
        if not merge_occurred:
            print(
                f"  WARN: Could not merge small chunk {current_chunk.get('_filename')}. Keeping as is."
            )
            final_chunks_list.append(current_chunk)

        # Always advance the loop counter
        i += 1

    # Return the list including the _filename key, it will be removed before saving
    return final_chunks_list


def save_chunks(chunks: list[dict], output_dir: str):
    """
    Saves the final list of processed chunks to the output directory.
    The filename is retrieved from the '_filename' key added during loading.
    """
    if not chunks:
        print("No chunks to save.")
        return

    print(f"\nSaving {len(chunks)} final chunks to: {output_dir}")
    saved_count = 0
    error_count = 0

    # Assume chunks still have the '_filename' key from loading
    for i, chunk_data in enumerate(chunks):
        # It's safer to use the index to generate a filename if _filename is missing
        # or rely on the structure (chap_sec_part) if present.
        # Let's assume _filename is present from load_and_sort_chunks.
        filename = chunk_data.get("_filename")
        if not filename:
            # Fallback filename generation if needed, though ideally _filename exists
            chap_num = chunk_data.get("chapter_number", 0)
            # Need a way to get section/part index if _filename is missing - complex.
            # For now, log error and skip.
            print(
                f"ERROR: Chunk data (index {i}) missing '_filename'. Cannot determine save path. Skipping."
            )
            error_count += 1
            continue

        output_filepath = os.path.join(output_dir, filename)
        # Create a copy to avoid modifying the list while iterating if pop is used later
        save_data = chunk_data.copy()
        save_data.pop("_filename", None)  # Remove internal key before saving

        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            saved_count += 1
        except (OSError, TypeError, Exception) as e:
            print(f"ERROR saving chunk {filename}: {e}")
            error_count += 1

    print(f"Successfully saved: {saved_count} chunks.")
    if error_count > 0:
        print(f"Failed to save: {error_count} chunks.")


def main():
    """
    Main execution function for Stage 5.

    Loads final chunks, performs a final merge pass for ultra-small chunks,
    and saves the results to the specified output directory.
    """
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    print("-" * 50)
    print("Running Stage 5: Final Merge of Ultra-Small Chunks")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Small chunk threshold: {ULTRA_SMALL_THRESHOLD} tokens")
    print(f"Max merged tokens    : {MAX_TOKENS}")
    print("-" * 50)

    # Ensure output directory exists
    create_directory(output_dir)

    # 1. Load and sort chunks from Stage 4 output
    loaded_chunks = load_and_sort_chunks(input_dir)
    if loaded_chunks is None:  # Check for critical loading error
        print("Exiting due to errors during chunk loading.")
        return
    if not loaded_chunks:  # Check if loading returned an empty list
        print("No chunks loaded. Nothing to merge.")
        return

    # 2. Perform the final merge pass
    final_chunks = merge_final_small_chunks(
        loaded_chunks, ULTRA_SMALL_THRESHOLD, MAX_TOKENS
    )
    print(f"\nTotal chunks after final merge pass: {len(final_chunks)}")

    # 3. Save the potentially modified chunks
    # 3. Save the potentially modified chunks
    # 4. Add the final 'sequence_number' field based on the sorted list
    for i, chunk in enumerate(final_chunks):
        chunk["sequence_number"] = i + 1 # Add 1-based sequence number field

    # 5. Save the final chunks using their original filenames (preserved in _filename)
    save_chunks(final_chunks, output_dir)

    print("-" * 50)
    print("Stage 5 finished.")
    print(f"Final output chunks are in: {output_dir}")
    print("-" * 50)


if __name__ == "__main__":
    main()
