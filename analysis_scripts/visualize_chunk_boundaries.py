#!/usr/bin/env python3
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import numpy as np # Import numpy for NaN handling

# Define directories
INPUT_DIR = "2E_final_merged_chunks" # UPDATED to Stage 5 output
OUTPUT_DIR = "visualizations" # Directory to save plots (will be removed)

# Ensure the output directory exists - REMOVED
# os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_filename(filename):
    """Extracts chapter, section, and part number from filename."""
    # Pattern: 00x_0xx_level_name_0xx.json OR chapNum_secIdx_..._partNum.json
    match = re.match(r'(\d+)_(\d+)_.*_(\d{3})\.json', filename)
    if match:
        chap_num = int(match.group(1))
        sec_idx = int(match.group(2)) # Original section index before splitting
        part_num = int(match.group(3))
        return chap_num, sec_idx, part_num
    else:
        # Fallback for potentially different naming, might need adjustment
        parts = filename.replace('.json', '').split('_')
        try:
            # Assuming format like ChapterX_SectionY_..._partZ
            chap_num = int(parts[0].replace('Chapter', '')) if 'Chapter' in parts[0] else int(parts[0])
            sec_idx = int(parts[1].replace('Section', '')) if 'Section' in parts[1] else int(parts[1])
            part_num = int(parts[-1]) if parts[-1].isdigit() else 0 # Assume 0 if no part number found
            return chap_num, sec_idx, part_num
        except (IndexError, ValueError):
            print(f"Warning: Could not parse filename format: {filename}")
            return None, None, None


def load_chunk_data(input_dir):
    """Loads data from all JSON chunk files into a pandas DataFrame."""
    all_data = []
    print(f"Loading data from: {input_dir}")
    try:
        filenames = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        if not filenames:
            print(f"Error: No JSON files found in {input_dir}")
            return None

        # Optional: Natural sort filenames if natsort is available
        try:
            import natsort
            filenames = natsort.natsorted(filenames)
        except ImportError:
            filenames.sort()
            print("Info: natsort not found, using standard sort for filenames.")

        for filename in filenames:
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                chap_num_file, sec_idx_file, part_num_file = parse_filename(filename)

                # Prefer data from JSON if available, fallback to filename parsing
                chap_num = data.get('chapter_number', chap_num_file)
                chap_name = data.get('chapter_name', f"Chapter {chap_num}" if chap_num is not None else "Unknown")
                token_count = data.get('chunk_token_count')
                part_num = data.get('chunk_part_number', part_num_file) # Prefer JSON part number
                page_start = data.get('section_page_start')
                page_end = data.get('section_page_end')

                if token_count is None:
                    print(f"Warning: 'chunk_token_count' missing in {filename}. Skipping.")
                    continue
                if chap_num is None:
                    print(f"Warning: Could not determine chapter number for {filename}. Skipping.")
                    continue

                all_data.append({
                    'filename': filename,
                    'chapter_number': chap_num,
                    'chapter_name': chap_name,
                    'section_index_file': sec_idx_file, # From filename, original section index
                    'part_number': part_num,
                    'token_count': token_count,
                    'page_start': page_start,
                    'page_end': page_end,
                })
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                # traceback.print_exc() # Uncomment for detailed traceback

        if not all_data:
            print("Error: No valid data loaded.")
            return None

        df = pd.DataFrame(all_data)
        # Ensure chapter_number is integer for sorting
        df['chapter_number'] = df['chapter_number'].astype(int)
        df = df.sort_values(by=['chapter_number', 'filename']).reset_index(drop=True)
        print(f"Successfully loaded data for {len(df)} chunks.")
        return df

    except FileNotFoundError:
        print(f"Error: Input directory not found: {input_dir}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        traceback.print_exc()
        return None


def create_visualizations(df, output_dir):
    """Generates and saves plots based on the DataFrame."""
    if df is None or df.empty:
        print("No data available for visualization.")
        return

    """Generates and displays plots based on the DataFrame, returns summary."""
    if df is None or df.empty:
        print("No data available for visualization.")
        return None

    print("\nGenerating visualizations...")
    sns.set_theme(style="whitegrid")

    # 1. Histogram of Chunk Token Sizes
    plt.figure(figsize=(10, 6))
    sns.histplot(df['token_count'], bins=50, kde=True)
    plt.title('Distribution of Chunk Token Sizes')
    plt.xlabel('Token Count per Chunk')
    plt.ylabel('Number of Chunks')
    plt.tight_layout()
    # plot_path = os.path.join(output_dir, 'chunk_token_size_distribution.png') # REMOVED
    # plt.savefig(plot_path) # REMOVED
    plt.show() # ADDED for inline display
    # plt.close() # REMOVED
    # print(f"Saved: {plot_path}") # REMOVED

    # --- Aggregate data per chapter ---
    chapter_stats = df.groupby(['chapter_number', 'chapter_name']).agg(
        total_chunks=('filename', 'count'),
        total_tokens=('token_count', 'sum'),
        avg_token_per_chunk=('token_count', 'mean'),
        min_token_per_chunk=('token_count', 'min'),
        max_token_per_chunk=('token_count', 'max')
    ).reset_index().sort_values('chapter_number')

    # 2. Number of Chunks per Chapter
    plt.figure(figsize=(12, 7))
    # Use chapter_name for labels but sort by chapter_number implicitly via chapter_stats order
    sns.barplot(x='chapter_name', y='total_chunks', data=chapter_stats, palette='viridis')
    plt.title('Number of Final Chunks per Chapter')
    plt.xlabel('Chapter')
    plt.ylabel('Number of Chunks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plot_path = os.path.join(output_dir, 'chunks_per_chapter.png') # REMOVED
    # plt.savefig(plot_path) # REMOVED
    plt.show() # ADDED
    # plt.close() # REMOVED
    # print(f"Saved: {plot_path}") # REMOVED

    # 3. Total Tokens per Chapter
    plt.figure(figsize=(12, 7))
    sns.barplot(x='chapter_name', y='total_tokens', data=chapter_stats, palette='magma')
    plt.title('Total Tokens per Chapter')
    plt.xlabel('Chapter')
    plt.ylabel('Total Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plot_path = os.path.join(output_dir, 'tokens_per_chapter.png') # REMOVED
    # plt.savefig(plot_path) # REMOVED
    plt.show() # ADDED
    # plt.close() # REMOVED
    # print(f"Saved: {plot_path}") # REMOVED

    # 4. Box Plot of Token Sizes per Chapter
    plt.figure(figsize=(12, 7))
    # Order by chapter number for consistency
    sorted_chapters = df.sort_values('chapter_number')['chapter_name'].unique()
    sns.boxplot(x='chapter_name', y='token_count', data=df, order=sorted_chapters, palette='coolwarm')
    plt.title('Distribution of Chunk Token Sizes per Chapter')
    plt.xlabel('Chapter')
    plt.ylabel('Token Count per Chunk')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 5. Average Pages Spanned per Chunk per Chapter
    # Calculate pages spanned, handle potential None values
    df['pages_spanned'] = df.apply(lambda row: row['page_end'] - row['page_start'] + 1 if pd.notna(row['page_start']) and pd.notna(row['page_end']) else np.nan, axis=1)

    if df['pages_spanned'].notna().any(): # Check if there's any valid page span data
        avg_pages_stats = df.dropna(subset=['pages_spanned']).groupby(['chapter_number', 'chapter_name'])['pages_spanned'].mean().reset_index().sort_values('chapter_number')

        plt.figure(figsize=(12, 7))
        sns.barplot(x='chapter_name', y='pages_spanned', data=avg_pages_stats, palette='crest')
        plt.title('Average Pages Spanned per Chunk per Chapter')
        plt.xlabel('Chapter')
        plt.ylabel('Average Pages Spanned')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping 'Average Pages Spanned' plot: No valid page start/end data found.")


    # --- Print Summary Table ---
    # Add page span info to summary if available
    if df['pages_spanned'].notna().any():
        page_summary = df.dropna(subset=['pages_spanned']).groupby(['chapter_number', 'chapter_name']).agg(
            avg_pages_spanned=('pages_spanned', 'mean'),
            min_pages_spanned=('pages_spanned', 'min'),
            max_pages_spanned=('pages_spanned', 'max')
        ).reset_index()
        chapter_stats = pd.merge(chapter_stats, page_summary, on=['chapter_number', 'chapter_name'], how='left')
        # Fill NaN for chapters that might not have had page numbers
        chapter_stats[['avg_pages_spanned', 'min_pages_spanned', 'max_pages_spanned']] = chapter_stats[['avg_pages_spanned', 'min_pages_spanned', 'max_pages_spanned']].fillna(0)
        # Format new columns
        chapter_stats['avg_pages_spanned'] = chapter_stats['avg_pages_spanned'].round(1)
        chapter_stats['min_pages_spanned'] = chapter_stats['min_pages_spanned'].astype(int)
        chapter_stats['max_pages_spanned'] = chapter_stats['max_pages_spanned'].astype(int)


    print("\n--- Summary Statistics per Chapter ---")
    # Format for better readability
    chapter_stats['avg_token_per_chunk'] = chapter_stats['avg_token_per_chunk'].round(1)
    # Select and rename columns for display
    summary_table = chapter_stats[[
        'chapter_number', 'chapter_name', 'total_chunks', 'total_tokens',
        'avg_token_per_chunk', 'min_token_per_chunk', 'max_token_per_chunk'
    ]].rename(columns={
        'chapter_number': 'Chap Num',
        'chapter_name': 'Chapter Name',
        'total_chunks': 'Chunks',
        'total_tokens': 'Total Tokens',
        'avg_token_per_chunk': 'Avg Tokens/Chunk',
        'min_token_per_chunk': 'Min Tokens/Chunk',
        'max_token_per_chunk': 'Max Tokens/Chunk',
        # Add new page columns if they exist
        **({'avg_pages_spanned': 'Avg Pages/Chunk',
            'min_pages_spanned': 'Min Pages/Chunk',
            'max_pages_spanned': 'Max Pages/Chunk'} if 'avg_pages_spanned' in chapter_stats.columns else {})
    })
    # Select columns in desired order, handling missing page columns
    display_cols = ['Chap Num', 'Chapter Name', 'Chunks', 'Total Tokens', 'Avg Tokens/Chunk', 'Min Tokens/Chunk', 'Max Tokens/Chunk']
    if 'Avg Pages/Chunk' in summary_table.columns:
        display_cols.extend(['Avg Pages/Chunk', 'Min Pages/Chunk', 'Max Pages/Chunk'])
    summary_table = summary_table[display_cols]


    # Print summary table to console (useful even in notebooks)
    print(summary_table.to_string(index=False))
    print("-" * 36)

    # Save summary table to CSV - REMOVED
    # csv_path = os.path.join(output_dir, 'chapter_summary_stats.csv')
    # try:
    #     summary_table.to_csv(csv_path, index=False)
    #     print(f"Saved summary table: {csv_path}")
    # except Exception as e:
    #     print(f"Error saving summary CSV: {e}")

    print("\nVisualizations generated and displayed.")
    return summary_table # Return the summary dataframe


def main(input_dir=INPUT_DIR): # Allow overriding input dir
    """Main function to load data and create visualizations."""
    print("-" * 50)
    print("Running Chunk Visualization Script")
    print(f"Input directory (chunks): {input_dir}") # Use function argument
    # print(f"Output directory (plots): {OUTPUT_DIR}") # REMOVED
    print("-" * 50)

    # Check for dependencies
    try:
        import pandas
        import matplotlib
        import seaborn
    except ImportError as e:
        print(f"\nError: Missing required library: {e.name}")
        print("Please install the required libraries:")
        print("pip install pandas matplotlib seaborn natsort") # Added natsort here
        return

    # Load data
    # Load data
    df_chunks = load_chunk_data(input_dir) # Use function argument

    # Create visualizations and get summary table
    summary_df = create_visualizations(df_chunks, None) # Pass None for output_dir

    # Optionally, you could return the dataframes from main if needed elsewhere
    # return df_chunks, summary_df

    print("-" * 50)
    print("Script finished.")
    print("-" * 50)


if __name__ == "__main__":
    main()
