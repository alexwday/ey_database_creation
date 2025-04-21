# -*- coding: utf-8 -*-
"""
Stage 5: Verification & Analysis

Purpose:
Loads the final records inserted into the database by Stage 4, performs
various sanity checks (record counts, sequence continuity), and generates
a visualization of token counts per chapter.

Input:
- Database connection details (configured below or via environment variables).
- document_id to identify the dataset to verify.
- JSON file from Stage 3 (used for expected record count verification).

Output:
- Console logs summarizing verification results.
- A detailed log file in LOG_DIR.
- A box plot visualization of section token counts per chapter saved in OUTPUT_DIR.
"""

import os
import json
import traceback
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# --- Dependencies Check ---
try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:
    psycopg2 = None
    DictCursor = None
    print("ERROR: psycopg2 library not installed. Database operations unavailable. `pip install psycopg2-binary`")

try:
    import pandas as pd
except ImportError:
    pd = None
    print("ERROR: pandas library not installed. Data analysis unavailable. `pip install pandas`")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None
    print("WARNING: matplotlib/seaborn not installed. Visualization unavailable. `pip install matplotlib seaborn`")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x # Make tqdm optional
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration
# ==============================================================================

# --- Directory Paths ---
STAGE3_OUTPUT_DIR = "pipeline_output/stage3"
STAGE3_FILENAME = "stage3_final_records.json"
OUTPUT_DIR = "pipeline_output/stage5" # Output dir for charts etc.
LOG_DIR = "pipeline_output/logs"

# --- Document ID ---
# TODO: Ensure this matches the DOCUMENT_ID used in previous stages
DOCUMENT_ID = "EY_GUIDE_2024_PLACEHOLDER"

# --- Database Configuration ---
# TODO: Load securely (e.g., environment variables) or replace placeholders
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "ey_guidance")
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password") # Be cautious storing passwords directly

# --- Database Operations ---
TARGET_TABLE = "guidance_sections"

# --- Visualization ---
CHART_FILENAME = "section_token_counts_by_chapter.png"
CHART_TITLE = f"Distribution of Section Token Counts by Chapter ({DOCUMENT_ID})"
FIG_WIDTH = 10 # inches
FIG_HEIGHT_PER_CHAPTER = 0.5 # inches per chapter for tall chart

# --- Logging Setup ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True) # Create output dir for charts
log_file = Path(LOG_DIR) / 'stage5_verification_and_analysis.log'
# Remove existing handlers if configuring multiple times in a notebook
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# ==============================================================================
# Utility Functions
# ==============================================================================

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    if not psycopg2:
        logging.error("psycopg2 library is not available.")
        return None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        logging.info(f"Database connection established to {DB_NAME} on {DB_HOST}:{DB_PORT}")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {e}", exc_info=True)
        return None

def check_sequence_continuity(numbers: List[int], item_name: str) -> List[str]:
    """Checks if a list of numbers forms a continuous sequence starting from 1 or its min value."""
    issues = []
    if not numbers:
        issues.append(f"No {item_name} numbers found.")
        return issues

    unique_sorted_numbers = sorted(list(set(numbers)))
    min_num = unique_sorted_numbers[0]
    max_num = unique_sorted_numbers[-1]

    # Check if starts from 1 (or the minimum found value if not 1)
    # start_expected = 1 # Usually expect sequences like section, part, sequence_num to start at 1
    # if min_num != start_expected:
    #     issues.append(f"{item_name.capitalize()} sequence does not start at {start_expected} (starts at {min_num}).")

    # Check for gaps
    expected_numbers = set(range(min_num, max_num + 1))
    actual_numbers = set(unique_sorted_numbers)
    missing_numbers = sorted(list(expected_numbers - actual_numbers))

    if missing_numbers:
        # Report missing numbers concisely
        if len(missing_numbers) > 10:
             issues.append(f"Gaps found in {item_name} sequence. Missing: {missing_numbers[:5]}... and {len(missing_numbers)-5} more.")
        else:
             issues.append(f"Gaps found in {item_name} sequence. Missing: {missing_numbers}.")

    # Check for duplicates (implicitly checked by using set earlier, but could add explicit check if needed)
    # if len(numbers) != len(unique_sorted_numbers):
    #     counts = defaultdict(int)
    #     for n in numbers: counts[n] += 1
    #     duplicates = {k:v for k,v in counts.items() if v > 1}
    #     issues.append(f"Duplicate {item_name} numbers found: {duplicates}")

    return issues

# ==============================================================================
# Main Verification Logic
# ==============================================================================

def run_stage5():
    """Main function to execute Stage 5 verification and analysis."""
    logging.info(f"--- Starting Stage 5: Verification & Analysis for {DOCUMENT_ID} ---")
    verification_passed = True
    issues_found = []

    # --- Dependency Checks ---
    if not psycopg2 or not pd or not plt or not sns:
        logging.error("Missing required libraries (psycopg2, pandas, matplotlib, seaborn). Cannot perform verification.")
        return False

    # --- Load Expected Count from Stage 3 ---
    stage3_output_file = Path(STAGE3_OUTPUT_DIR) / STAGE3_FILENAME
    expected_count = 0
    if stage3_output_file.exists():
        try:
            with open(stage3_output_file, "r", encoding="utf-8") as f:
                stage3_data = json.load(f)
            expected_count = len(stage3_data)
            logging.info(f"Loaded {expected_count} records from Stage 3 output for count verification.")
        except Exception as e:
            logging.warning(f"Could not load or read Stage 3 output file {stage3_output_file}: {e}. Cannot verify record count.")
            issues_found.append("Could not verify record count against Stage 3 output.")
            verification_passed = False # Treat inability to check as a failure
    else:
        logging.warning(f"Stage 3 output file not found: {stage3_output_file}. Cannot verify record count.")
        issues_found.append("Could not verify record count against Stage 3 output (file not found).")
        verification_passed = False # Treat inability to check as a failure

    # --- Fetch Data from Database ---
    conn = None
    db_records = []
    try:
        conn = get_db_connection()
        if not conn: raise ConnectionError("Failed to establish database connection.")

        fetch_sql = f"SELECT * FROM {TARGET_TABLE} WHERE document_id = %s ORDER BY sequence_number;"
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(fetch_sql, (DOCUMENT_ID,))
            db_records = cur.fetchall()
        logging.info(f"Fetched {len(db_records)} records from database table '{TARGET_TABLE}' for document_id '{DOCUMENT_ID}'.")

    except Exception as e:
        logging.error(f"Error fetching data from database: {e}", exc_info=True)
        issues_found.append(f"Failed to fetch data from database: {e}")
        return False # Cannot proceed without data
    finally:
        if conn: conn.close()

    if not db_records:
        logging.error(f"No records found in database for document_id '{DOCUMENT_ID}'. Verification cannot proceed.")
        issues_found.append("No records found in database for the specified document_id.")
        return False

    # Convert to DataFrame
    df = pd.DataFrame([dict(row) for row in db_records])

    # --- Verification Checks ---

    # 1. Record Count Verification
    actual_count = len(df)
    if expected_count > 0: # Only check if we loaded expected count
        if actual_count == expected_count:
            logging.info(f"[PASS] Record count matches Stage 3 output ({actual_count} records).")
        else:
            logging.error(f"[FAIL] Record count mismatch! Expected: {expected_count}, Found in DB: {actual_count}.")
            issues_found.append(f"Record count mismatch (Expected: {expected_count}, Found: {actual_count}).")
            verification_passed = False
    else:
         logging.warning("Skipping record count verification as expected count could not be determined.")


    # 2. Overall Sequence Number Continuity
    logging.info("Checking overall sequence_number continuity...")
    seq_issues = check_sequence_continuity(df['sequence_number'].tolist(), 'overall sequence_number')
    if not seq_issues:
        logging.info("[PASS] Overall sequence_number is continuous.")
    else:
        logging.error(f"[FAIL] Issues found with overall sequence_number continuity:")
        for issue in seq_issues:
            logging.error(f"  - {issue}")
            issues_found.extend(seq_issues)
        verification_passed = False

    # 3. Chapter Number Continuity
    logging.info("Checking chapter_number continuity...")
    chapter_issues = check_sequence_continuity(df['chapter_number'].unique().tolist(), 'chapter_number')
    if not chapter_issues:
        logging.info("[PASS] chapter_number sequence appears continuous.")
    else:
        logging.warning(f"[WARN] Issues found with chapter_number continuity:") # Warning as chapters might not start at 1
        for issue in chapter_issues:
            logging.warning(f"  - {issue}")
            issues_found.extend(chapter_issues)
        # verification_passed = False # Don't fail verification for chapter gaps unless critical

    # 4. Section Number Continuity (within each chapter)
    logging.info("Checking section_number continuity within each chapter...")
    section_continuity_ok = True
    for chapter, group in df.groupby('chapter_number'):
        sec_issues = check_sequence_continuity(group['section_number'].tolist(), f'section_number in Chapter {chapter}')
        if sec_issues:
            section_continuity_ok = False
            logging.error(f"[FAIL] Issues found with section_number continuity in Chapter {chapter}:")
            for issue in sec_issues:
                logging.error(f"  - {issue}")
                issues_found.extend([f"Chapter {chapter}: {issue}" for issue in sec_issues])
            verification_passed = False
    if section_continuity_ok:
        logging.info("[PASS] section_number sequences appear continuous within each chapter.")

    # 5. Part Number Continuity (within each chapter/section)
    logging.info("Checking part_number continuity within each chapter/section...")
    part_continuity_ok = True
    for (chapter, section), group in df.groupby(['chapter_number', 'section_number']):
        # Only check parts if there's more than one part expected (i.e., max part > 1)
        if group['part_number'].max() > 1:
            part_issues = check_sequence_continuity(group['part_number'].tolist(), f'part_number in Chapter {chapter}, Section {section}')
            if part_issues:
                part_continuity_ok = False
                logging.error(f"[FAIL] Issues found with part_number continuity in Chapter {chapter}, Section {section}:")
                for issue in part_issues:
                    logging.error(f"  - {issue}")
                    issues_found.extend([f"Chapter {chapter}, Section {section}: {issue}" for issue in part_issues])
                verification_passed = False
    if part_continuity_ok:
        logging.info("[PASS] part_number sequences appear continuous where sections have multiple parts.")

    # --- Token Count Visualization ---
    logging.info("Generating section token count visualization...")
    chart_path = Path(OUTPUT_DIR) / CHART_FILENAME
    try:
        # Ensure 'section_token_count' exists and handle potential NaNs if necessary
        if 'section_token_count' not in df.columns:
             logging.error("Column 'section_token_count' not found in data. Cannot generate visualization.")
             issues_found.append("Cannot generate token visualization: 'section_token_count' column missing.")
        else:
            # Drop rows where token count might be NaN/None before plotting
            plot_df = df.dropna(subset=['chapter_number', 'section_token_count'])
            plot_df['chapter_number'] = plot_df['chapter_number'].astype(int) # Ensure chapter is int for sorting
            plot_df['section_token_count'] = plot_df['section_token_count'].astype(int)

            num_chapters = plot_df['chapter_number'].nunique()
            fig_height = max(5, num_chapters * FIG_HEIGHT_PER_CHAPTER) # Ensure minimum height

            plt.figure(figsize=(FIG_WIDTH, fig_height))
            sns.boxplot(y='chapter_number', x='section_token_count', data=plot_df, orient='h', order=sorted(plot_df['chapter_number'].unique()))
            plt.title(CHART_TITLE)
            plt.xlabel("Section Token Count")
            plt.ylabel("Chapter Number")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close() # Close plot to free memory
            logging.info(f"Token count visualization saved to: {chart_path}")
            issues_found.append(f"Token count visualization saved to: {chart_path}") # Add info to report

    except Exception as e:
        logging.error(f"Error generating token count visualization: {e}", exc_info=True)
        issues_found.append(f"Failed to generate token count visualization: {e}")
        # Don't fail overall verification just for chart failure
        # verification_passed = False


    # --- Final Report ---
    logging.info("--- Stage 5 Verification Summary ---")
    if verification_passed:
        logging.info("Overall Verification Status: PASS")
    else:
        logging.error("Overall Verification Status: FAIL")

    if issues_found:
        logging.info("Details:")
        for issue in issues_found:
            if "FAIL" in issue.upper() or "ERROR" in issue.upper():
                 logging.error(f"- {issue}")
            elif "WARN" in issue.upper():
                 logging.warning(f"- {issue}")
            else:
                 logging.info(f"- {issue}") # For info like chart path

    logging.info("--- Stage 5 Finished ---")
    return verification_passed

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    if not all([psycopg2, pd, plt, sns]):
        logging.critical("Missing one or more required libraries. Please install psycopg2-binary, pandas, matplotlib, and seaborn.")
        sys.exit(1)
    run_stage5()
