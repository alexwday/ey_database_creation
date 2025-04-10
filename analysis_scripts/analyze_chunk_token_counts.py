#!/usr/bin/env python3
import os
import json
import pandas as pd
import natsort
from collections import defaultdict
import traceback

# Define the directory containing the final merged chunks
INPUT_DIR = "2E_final_merged_chunks"

def load_and_group_chunks(input_dir):
    """Loads chunk data and groups it by chapter and section hierarchy."""
    grouped_chunks = defaultdict(lambda: defaultdict(list))
    filenames = []
    print(f"Loading chunks from: {input_dir}")
    try:
        filenames = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        if not filenames:
            print(f"Error: No JSON files found in {input_dir}")
            return None
        filenames = natsort.natsorted(filenames) # Ensure consistent order
        print(f"Found {len(filenames)} chunk files.")
    except FileNotFoundError:
        print(f"Error: Input directory not found: {input_dir}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred listing files: {e}")
        traceback.print_exc()
        return None

    loaded_count = 0
    error_count = 0
    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- Get Grouping Keys ---
            chapter_number = data.get('chapter_number')
            # Use section_hierarchy if available, fallback to section_title or filename
            section_key = data.get('section_hierarchy')
            if section_key is None:
                section_key = data.get('section_title', f"Unknown_Section_{filename}")

            if chapter_number is None:
                print(f"Warning: Missing 'chapter_number' in {filename}. Skipping.")
                error_count += 1
                continue

            # --- Get Token Count ---
            token_count = data.get('chunk_token_count')
            if token_count is None:
                print(f"Warning: Missing 'chunk_token_count' in {filename}. Skipping.")
                error_count += 1
                continue

            grouped_chunks[chapter_number][section_key].append(token_count)
            loaded_count += 1

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}. Skipping.")
            error_count += 1
        except Exception as e:
            print(f"Error processing file {filename}: {e}. Skipping.")
            traceback.print_exc()
            error_count += 1

    print(f"Successfully processed {loaded_count} chunks.")
    if error_count > 0:
        print(f"Skipped {error_count} chunks due to errors.")

    if not grouped_chunks:
        print("No chunks were successfully loaded and grouped.")
        return None

    return grouped_chunks

def calculate_token_counts(grouped_chunks):
    """Calculates token counts per section and chapter."""
    section_data = []
    chapter_totals = defaultdict(int)

    # Sort chapters naturally
    sorted_chapters = natsort.natsorted(grouped_chunks.keys())

    for chapter_number in sorted_chapters:
        chapter_sections = grouped_chunks[chapter_number]
        # Sort sections naturally within the chapter
        sorted_sections = natsort.natsorted(chapter_sections.keys())

        for section_key in sorted_sections:
            section_token_list = chapter_sections[section_key]
            section_total_tokens = sum(section_token_list)

            section_data.append({
                'chapter_number': chapter_number,
                'section_identifier': section_key, # Using 'identifier' as it could be hierarchy or title
                'section_token_count': section_total_tokens,
                'chunk_count': len(section_token_list)
            })
            chapter_totals[chapter_number] += section_total_tokens

    # Convert chapter totals to a more usable format (e.g., DataFrame or dict)
    chapter_summary = [{'chapter_number': k, 'total_chapter_tokens': v} for k, v in chapter_totals.items()]
    chapter_summary_df = pd.DataFrame(chapter_summary).sort_values(by='chapter_number').reset_index(drop=True)


    if not section_data:
        print("Warning: No section data generated.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames

    # Create DataFrame
    df = pd.DataFrame(section_data)
    return df, chapter_summary_df


def get_token_analysis_df(input_dir=INPUT_DIR):
    """
    Main function to load data, calculate token counts, and return DataFrames.
    Designed to be called from a notebook.

    Args:
        input_dir (str): Path to the directory containing chunk JSON files.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
               - sections_df: DataFrame with token counts per section.
               - chapters_df: DataFrame with total token counts per chapter.
               Returns (None, None) if loading fails.
    """
    print("-" * 50)
    print("Starting Token Count Analysis...")
    grouped_chunks = load_and_group_chunks(input_dir)

    if grouped_chunks is None:
        print("Failed to load or group chunks. Exiting analysis.")
        return None, None # Indicate failure

    sections_df, chapters_df = calculate_token_counts(grouped_chunks)
    print("Token count calculation complete.")
    print(f"Analyzed {chapters_df.shape[0]} chapters and {sections_df.shape[0]} sections.")
    print("-" * 50)
    return sections_df, chapters_df

if __name__ == "__main__":
    # Example usage when run as a script
    sections_df, chapters_df = get_token_analysis_df()

    if sections_df is not None and chapters_df is not None:
        print("\n--- Chapter Token Summary ---")
        print(chapters_df.to_string()) # Print full chapter summary

        print(f"\nTotal Chapters: {chapters_df.shape[0]}")
        print(f"Average Chapter Tokens: {chapters_df['total_chapter_tokens'].mean():.2f}")
        print(f"Median Chapter Tokens: {chapters_df['total_chapter_tokens'].median():.2f}")
        print(f"Max Chapter Tokens: {chapters_df['total_chapter_tokens'].max()}")
        print(f"Min Chapter Tokens: {chapters_df['total_chapter_tokens'].min()}")

        # Identify chapters potentially exceeding a limit (e.g., 80k)
        LIMIT = 80000
        over_limit = chapters_df[chapters_df['total_chapter_tokens'] > LIMIT]
        if not over_limit.empty:
            print(f"\nChapters potentially over {LIMIT} tokens:")
            print(over_limit[['chapter_number', 'total_chapter_tokens']].to_string(index=False))
        else:
            print(f"\nNo chapters appear to exceed {LIMIT} tokens.")


        print("\n--- Section Token Summary ---")
        print(f"Total Sections: {sections_df.shape[0]}")
        print(f"Average Section Tokens: {sections_df['section_token_count'].mean():.2f}")
        print(f"Median Section Tokens: {sections_df['section_token_count'].median():.2f}")
        print(f"Max Section Tokens: {sections_df['section_token_count'].max()}")
        print(f"Min Section Tokens: {sections_df['section_token_count'].min()}")

        # Save to CSV if needed
        # sections_df.to_csv("section_token_counts.csv", index=False)
        # chapters_df.to_csv("chapter_token_counts.csv", index=False)
        # print("\nSaved counts to section_token_counts.csv and chapter_token_counts.csv")
    else:
        print("Analysis could not be completed due to errors.")
