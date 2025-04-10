#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2: Section-Level Analysis

Goal: To generate more specific details for each section within a chapter,
leveraging the chapter-level context obtained in Phase 1 (7_generate_chapter_details.py).

Input:
- Chunk JSON files from CHUNK_INPUT_DIR (e.g., 2E_final_merged_chunks).
- Chapter details (summary, tags) JSON files from CHAPTER_DETAILS_INPUT_DIR (e.g., 3A_chapter_details).

Output:
- JSON files containing section details (summary, tags, standard, standard codes)
  for each section, saved to SECTION_DETAILS_OUTPUT_DIR (e.g., 3B_section_details).
"""

import os
import requests
import json
import time
import traceback
import natsort
import logging
from collections import defaultdict
from openai import OpenAI, APIError
from pathlib import Path
import tiktoken # Ensure tiktoken is installed

# --- Configuration ---
# Define BASE_DIR based on current working directory for notebook compatibility
# Assumes the notebook is run from the project root (/Users/alexwday/Projects/ey_database_creation)
BASE_DIR = Path.cwd()
CHUNK_INPUT_DIR = BASE_DIR / "data" / "2E_final_merged_chunks"
CHAPTER_DETAILS_INPUT_DIR = BASE_DIR / "data" / "3A_chapter_details"
SECTION_DETAILS_OUTPUT_DIR = BASE_DIR / "data" / "3B_section_details"
LOG_DIR = BASE_DIR / "logs"
LOG_LEVEL = logging.INFO

# --- Testing Configuration ---
PROCESS_CHAPTER = 1 # Set to a specific chapter number (e.g., 1) for testing, or None to process all

# --- Constants from Script 7 (Adapt as needed) ---
BASE_URL = "https://api.example.com/v1" # Replace with actual endpoint if different
MODEL_NAME = "gpt-4-turbo-nonp" # Or your preferred model
MAX_COMPLETION_TOKENS = 4096 # Max tokens for the *response* (adjust if needed)
TEMPERATURE = 0.5 # Adjust as needed

# OAuth settings - **LOAD SECURELY (e.g., environment variables)**
OAUTH_URL = "https://api.example.com/oauth/token" # Replace if needed
CLIENT_ID = os.environ.get("RBC_OAUTH_CLIENT_ID", "your_client_id") # Placeholder
CLIENT_SECRET = os.environ.get("RBC_OAUTH_CLIENT_SECRET", "your_client_secret") # Placeholder

# SSL certificate settings - **Ensure paths are correct**
SSL_SOURCE_PATH = os.environ.get("RBC_SSL_SOURCE_PATH", "/path/to/your/rbc-ca-bundle.cer") # Placeholder
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer" # Temporary local path

# Token cost settings (Optional, for tracking)
PROMPT_TOKEN_COST = 0.01    # Cost per 1K prompt tokens
COMPLETION_TOKEN_COST = 0.03 # Cost per 1K completion tokens

# --- Internal Constants ---
_SSL_CONFIGURED = False # Flag to avoid redundant SSL setup
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 5 # seconds
MAX_RECENT_SUMMARIES = 20 # Number of previous section summaries to include in context

# --- Tool Schema for Section Details ---
SECTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_section_details",
        "description": "Extracts detailed information about a specific document section based on its content and the overall chapter context.",
        "parameters": {
            "type": "object",
            "properties": {
                "section_summary": {
                    "type": "string",
                    "description": "A concise summary (1-3 sentences) capturing the core topic or purpose of this section, suitable for reranking search results."
                },
                "section_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of 5-15 granular keywords or tags specific to this section's content."
                },
                "section_standard": {
                    "type": "string",
                    "description": "The primary accounting or reporting standard applicable to this section (e.g., 'IFRS', 'US GAAP', 'N/A')."
                },
                "section_standard_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of specific standard codes explicitly mentioned or directly relevant in the section (e.g., ['IFRS 16', 'IAS 17'])."
                }
            },
            "required": ["section_summary", "section_tags", "section_standard", "section_standard_codes"]
        }
    }
}

# --- Logging Setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / '8_generate_section_details.log'
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("--- Starting Section-Level Detail Generation ---")


# --- Helper Functions (Copied/Adapted from 7_generate_chapter_details.py) ---

# Initialize tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logging.warning(f"Could not get tiktoken encoder 'cl100k_base': {e}. Using fallback length calculation.")
    tokenizer = None

def count_tokens(text):
    """Count tokens in text using the tokenizer."""
    if not text:
        return 0
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.warning(f"Tiktoken encoding failed: {e}. Falling back to len/4.")
            return len(text) // 4 # Fallback
    else:
        return len(text) // 4

def _setup_ssl(source_path=SSL_SOURCE_PATH, local_path=SSL_LOCAL_PATH):
    """Copies SSL cert locally and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True # Already configured

    logging.info("Setting up SSL certificate...")
    try:
        source = Path(source_path)
        local = Path(local_path)

        if not source.is_file():
            logging.error(f"SSL source certificate not found at {source_path}")
            return False

        local.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as source_file:
            content = source_file.read()
        with open(local, "wb") as dest_file:
            dest_file.write(content)

        os.environ["SSL_CERT_FILE"] = str(local)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local)
        logging.info(f"SSL certificate configured successfully at: {local}")
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        logging.error(f"Error setting up SSL certificate: {e}", exc_info=True)
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH):
    """Retrieves OAuth token from the specified endpoint."""
    logging.info("Attempting to get OAuth token...")
    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    try:
        response = requests.post(
            oauth_url,
            data=payload,
            timeout=30,
            verify=ssl_verify_path # Use the configured local path
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        token_data = response.json()
        oauth_token = token_data.get('access_token')
        if not oauth_token:
            logging.error("Error: 'access_token' not found in OAuth response.")
            return None
        logging.info("OAuth token obtained successfully.")
        return oauth_token
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting OAuth token: {e}", exc_info=True)
        return None

def create_openai_client(base_url=BASE_URL):
    """Sets up SSL, gets OAuth token, and creates the OpenAI client."""
    if not _setup_ssl():
        logging.error("Aborting client creation due to SSL setup failure.")
        return None # SSL setup failed

    api_key = _get_oauth_token()
    if not api_key:
        logging.error("Aborting client creation due to OAuth token failure.")
        return None # Token retrieval failed

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logging.info("OpenAI client created successfully.")
        return client
    except Exception as e:
        logging.error(f"Error creating OpenAI client: {e}", exc_info=True)
        return None

def _call_gpt_with_retry(client, model, messages, max_tokens, temperature, tools=None, tool_choice=None):
    """Makes the API call with retry logic, supporting tool calls."""
    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            logging.info(f"Making API call (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...")
            completion_kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if tools and tool_choice:
                completion_kwargs["tools"] = tools
                completion_kwargs["tool_choice"] = tool_choice
                logging.info("Making API call with tool choice...")
            else:
                # Defaulting to tool use for this script, but keeping structure
                logging.warning("API call initiated without explicit tool choice - this script expects tool use.")
                completion_kwargs["tools"] = tools # Still send tools
                # completion_kwargs["response_format"] = {"type": "json_object"} # Use if not using tools

            response = client.chat.completions.create(**completion_kwargs)
            logging.info("API call successful.")
            response_message = response.choices[0].message

            # --- Tool Call Handling ---
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                expected_tool_name = None
                if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                     expected_tool_name = tool_choice.get("function", {}).get("name")

                if not expected_tool_name:
                     logging.warning(f"Could not determine expected tool name from tool_choice: {tool_choice}")
                     # Assume the first tool call is the one we want if only one tool was provided
                     if tools and len(tools) == 1 and tools[0].get("type") == "function":
                         expected_tool_name = tools[0].get("function", {}).get("name")

                if expected_tool_name and tool_call.function.name != expected_tool_name:
                     raise ValueError(f"Expected tool '{expected_tool_name}' but received '{tool_call.function.name}'")
                elif not expected_tool_name:
                     logging.warning(f"Proceeding with received tool call '{tool_call.function.name}' as expected name wasn't determined.")


                function_args_json = tool_call.function.arguments
                return function_args_json, response.usage
            else:
                # Handle cases where the model might return content instead of tool call
                fallback_content = response_message.content
                logging.warning(f"Model did not return tool calls as expected. Content: {fallback_content[:200]}...")
                # Attempt to parse content as JSON if it looks like it
                if fallback_content and fallback_content.strip().startswith('{'):
                    logging.info("Attempting to parse fallback content as JSON.")
                    return fallback_content, response.usage # Let the parser handle it
                else:
                    raise ValueError("Expected tool calls in response, but none found and content is not JSON.")

        except APIError as e:
            logging.warning(f"API Error on attempt {attempt + 1}: {e}")
            last_exception = e
            time.sleep(_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logging.warning(f"Non-API Error on attempt {attempt + 1}: {e}", exc_info=True)
            last_exception = e
            time.sleep(_RETRY_DELAY)

    logging.error(f"API call failed after {_RETRY_ATTEMPTS} attempts.")
    if last_exception:
        raise last_exception
    else:
        raise Exception("API call failed for unknown reasons.")


def parse_gpt_json_response(response_content_str, expected_keys):
    """Parses JSON response from GPT and validates expected keys."""
    try:
        # Handle potential markdown code blocks
        if response_content_str.strip().startswith("```json"):
            response_content_str = response_content_str.strip()[7:-3].strip()
        elif response_content_str.strip().startswith("```"):
             response_content_str = response_content_str.strip()[3:-3].strip()

        data = json.loads(response_content_str)
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object.")

        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            # Allow section_standard_codes to be missing or empty list initially
            if not ("section_standard_codes" in missing_keys and len(missing_keys) == 1):
                 raise ValueError(f"Missing expected keys in response: {', '.join(missing_keys)}")
            else:
                 logging.warning("Key 'section_standard_codes' missing, will default to empty list.")
                 data['section_standard_codes'] = [] # Add default empty list


        # Type checks (optional but good practice)
        if 'section_summary' in expected_keys and not isinstance(data.get('section_summary'), str):
             logging.warning("Type mismatch: 'section_summary' is not a string.")
        if 'section_tags' in expected_keys and not isinstance(data.get('section_tags'), list):
             logging.warning("Type mismatch: 'section_tags' is not a list.")
        if 'section_standard' in expected_keys and not isinstance(data.get('section_standard'), str):
             logging.warning("Type mismatch: 'section_standard' is not a string.")
        if 'section_standard_codes' in expected_keys and not isinstance(data.get('section_standard_codes'), list):
             logging.warning("Type mismatch: 'section_standard_codes' is not a list.")


        logging.info("GPT JSON response parsed successfully.")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding GPT JSON response: {e}")
        logging.error(f"Raw response string: {response_content_str[:500]}...")
        return None
    except ValueError as e:
        logging.error(f"Error validating GPT JSON response: {e}")
        logging.error(f"Raw response string: {response_content_str[:500]}...")
        return None

def load_all_chunk_data_grouped(input_dir=CHUNK_INPUT_DIR):
    """
    Loads all chunk JSONs from the input directory, validates required fields
    ('order', 'chapter_number', 'content', 'chunk_token_count', section identifier),
    sorts them by 'order', and groups them by 'chapter_number'.
    Returns a dictionary mapping chapter numbers to lists of chunk dicts.
    """
    all_chunks_data = []
    input_path = Path(input_dir)
    logging.info(f"Loading chunks from: {input_path}")

    if not input_path.is_dir():
        logging.error(f"Input directory not found: {input_path}")
        return None

    filenames = [f for f in input_path.iterdir() if f.is_file() and f.suffix == ".json"]
    if not filenames:
        logging.warning(f"No JSON files found in {input_path}")
        return None
    logging.info(f"Found {len(filenames)} chunk files. Loading data...")

    loaded_count = 0
    error_count = 0
    skipped_missing_field = 0

    for filepath in filenames:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- Validation ---
            required_fields = ['order', 'chapter_number', 'content', 'chunk_token_count']
            # Use 'section_hierarchy' first, fallback to 'section_title'
            section_id_field = 'section_hierarchy' if 'section_hierarchy' in data else 'section_title'
            if section_id_field not in data:
                 logging.warning(f"Missing 'section_hierarchy' or 'section_title' in {filepath.name}. Skipping.")
                 skipped_missing_field += 1
                 continue
            required_fields.append(section_id_field) # Add the found section identifier field

            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                logging.warning(f"Missing required fields {missing_fields} in {filepath.name}. Skipping.")
                skipped_missing_field += 1
                continue

            if not isinstance(data['order'], int):
                 logging.warning(f"Invalid 'order' field type in {filepath.name}. Skipping.")
                 skipped_missing_field += 1
                 continue
            # --- End Validation ---

            # Store which field was used for section ID for consistency later
            data['_section_id_field'] = section_id_field
            data['_section_id_value'] = data[section_id_field]

            all_chunks_data.append(data)
            loaded_count += 1

        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from {filepath.name}. Skipping.")
            error_count += 1
        except Exception as e:
            logging.error(f"Error processing file {filepath.name}: {e}. Skipping.", exc_info=True)
            error_count += 1

    logging.info(f"Successfully loaded data for {loaded_count} chunks.")
    if skipped_missing_field > 0:
        logging.warning(f"Skipped {skipped_missing_field} chunks missing required fields.")
    if error_count > 0:
        logging.warning(f"Skipped {error_count} chunks due to other errors.")

    if not all_chunks_data:
        logging.error("No valid chunks were loaded.")
        return None

    # --- Sort by the 'order' field ---
    try:
        all_chunks_data.sort(key=lambda x: x['order'])
        logging.info(f"Successfully sorted {len(all_chunks_data)} chunks by 'order' field.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during sorting: {e}", exc_info=True)
        return None # Cannot proceed without sorting

    # --- Group by chapter_number ---
    grouped_by_chapter = defaultdict(list)
    for chunk in all_chunks_data:
        grouped_by_chapter[chunk['chapter_number']].append(chunk)

    # --- Sort chapters naturally ---
    try:
        if natsort:
            sorted_grouped_by_chapter = dict(natsort.natsorted(grouped_by_chapter.items()))
            logging.info(f"Grouped data into {len(sorted_grouped_by_chapter)} chapters (naturally sorted).")
        else:
            sorted_grouped_by_chapter = dict(sorted(grouped_by_chapter.items()))
            logging.info(f"Grouped data into {len(sorted_grouped_by_chapter)} chapters (standard sort).")
    except Exception as e:
        logging.warning(f"Could not sort chapter keys: {e}. Returning unsorted chapters.")
        sorted_grouped_by_chapter = dict(grouped_by_chapter) # Return unsorted if error

    return sorted_grouped_by_chapter

def load_chapter_details(chapter_number, input_dir=CHAPTER_DETAILS_INPUT_DIR):
    """Loads the pre-generated summary and tags for a specific chapter."""
    details_path = Path(input_dir) / f"chapter_{chapter_number}_details.json"
    if not details_path.exists():
        logging.warning(f"Chapter details file not found for chapter {chapter_number} at {details_path}")
        return None
    try:
        with open(details_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Basic validation
        if 'chapter_summary' not in data or 'chapter_tags' not in data:
             logging.warning(f"Chapter details file {details_path} is missing 'chapter_summary' or 'chapter_tags'.")
             return None
        return data
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from chapter details file: {details_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading chapter details file {details_path}: {e}", exc_info=True)
        return None

def group_chunks_by_section(chapter_chunks):
    """Groups chunks within a chapter by their section identifier."""
    grouped_by_section = defaultdict(list)
    if not chapter_chunks:
        return grouped_by_section

    # Assume all chunks in the list use the same section ID field determined during loading
    section_id_field = chapter_chunks[0].get('_section_id_field', 'section_hierarchy') # Default fallback

    for chunk in chapter_chunks:
        section_id = chunk.get(section_id_field, 'Unknown Section')
        grouped_by_section[section_id].append(chunk)

    logging.debug(f"Grouped {len(chapter_chunks)} chunks into {len(grouped_by_section)} sections using field '{section_id_field}'.")
    return grouped_by_section


# --- Phase 2: Section Level Functions ---

def _build_section_prompt(section_text, chapter_summary, chapter_tags, previous_section_summaries=None):
    """Builds the messages list for the section processing call."""
    if previous_section_summaries is None:
        previous_section_summaries = []

    system_prompt = """<role>You are an expert financial reporting specialist.</role>
<source_material>You are analyzing a specific section within a chapter from an EY technical accounting guidance manual. You are provided with the overall chapter summary/tags and summaries of recently processed sections from the same chapter.</source_material>
<task>Your primary task is to generate a **concise summary (1-3 sentences)** for the current section, suitable for use in reranking search results. Additionally, extract relevant tags, the primary applicable accounting standard, and specific standard codes mentioned. Use the 'extract_section_details' tool for your response.</task>
<guardrails>Base your analysis strictly on the provided section text and context. Focus on capturing the core topic/purpose concisely for the summary. Ensure tags and standard codes are precise and derived from the section text.</guardrails>"""

    user_prompt_elements = ["<prompt>"]
    user_prompt_elements.append("<style>Concise, factual, keyword-focused for summary; technical and precise for other fields.</style>")
    user_prompt_elements.append("<tone>Professional, objective, expert.</tone>")
    user_prompt_elements.append("<audience>Accounting professionals needing specific guidance on this section.</audience>")
    user_prompt_elements.append('<response_format>Use the "extract_section_details" tool.</response_format>')

    user_prompt_elements.append("<overall_chapter_context>")
    user_prompt_elements.append(f"<chapter_summary>{chapter_summary}</chapter_summary>")
    user_prompt_elements.append(f"<chapter_tags>{json.dumps(chapter_tags)}</chapter_tags>")
    user_prompt_elements.append("</overall_chapter_context>")

    if previous_section_summaries:
        user_prompt_elements.append("<recent_section_context>")
        for i, summary in enumerate(previous_section_summaries):
            user_prompt_elements.append(f"<previous_section_{i+1}_summary>{summary}</previous_section_{i+1}_summary>")
        user_prompt_elements.append("</recent_section_context>")

    user_prompt_elements.append(f"<current_section_text>{section_text}</current_section_text>")

    user_prompt_elements.append("<instructions>")
    user_prompt_elements.append("""
    **Analysis Objective:** Analyze the provided <current_section_text> considering the <overall_chapter_context> and <recent_section_context> (if provided).
    **Action:** Generate the following details for the **current section** using the 'extract_section_details' tool:
    1.  **section_summary:** A **very concise summary (1-3 sentences)** capturing the core topic or purpose of this section. This summary will be used to help rerank search results, so it should be distinct and informative at a glance.
    2.  **section_tags:** 5-15 granular tags specific to THIS SECTION's content.
    3.  **section_standard:** Identify the single, primary accounting standard framework most relevant to THIS SECTION (e.g., 'IFRS', 'US GAAP', 'N/A').
    4.  **section_standard_codes:** List ALL specific standard codes (e.g., 'IFRS 16', 'IAS 36.12', 'ASC 842-10-15') explicitly mentioned within THIS SECTION's text. Provide an empty list [] if none are mentioned.
    """)
    user_prompt_elements.append("</instructions>")
    user_prompt_elements.append("</prompt>")
    user_prompt = "\n".join(user_prompt_elements)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def process_section(section_id, section_chunks, chapter_details, previous_section_summaries, client, model_name, max_completion_tokens, temperature):
    """Processes a single section using GPT, including context from previous sections."""
    logging.info(f"Processing section: {section_id} with {len(previous_section_summaries)} previous summaries in context.")

    if not section_chunks:
        logging.warning(f"No chunks provided for section {section_id}. Skipping.")
        return None

    # Reconstruct section text (assuming chunks are sorted by 'order')
    section_text = "\n\n".join([chunk.get('content', '') for chunk in section_chunks])
    section_token_count = sum(chunk.get('chunk_token_count', 0) for chunk in section_chunks)
    logging.info(f"Section '{section_id}' - Estimated tokens: {section_token_count}")

    # Basic check - might need segmentation like chapters if sections get too large
    # TODO: Implement section segmentation if necessary, similar to get_chapter_level_details
    # processing_limit = INPUT_TOKEN_LIMIT - max_completion_tokens - TOKEN_BUFFER # Define these constants if needed
    # if section_token_count > processing_limit:
    #     logging.warning(f"Section {section_id} ({section_token_count} tokens) may exceed processing limit. Segmentation not yet implemented.")
        # return None # Or implement segmentation

    messages = _build_section_prompt(
        section_text=section_text,
        chapter_summary=chapter_details.get('chapter_summary', 'Summary not available.'),
        chapter_tags=chapter_details.get('chapter_tags', []),
        previous_section_summaries=previous_section_summaries
    )

    prompt_tokens_est = sum(count_tokens(msg["content"]) for msg in messages)
    logging.info(f"Estimated prompt tokens for section '{section_id}': {prompt_tokens_est}")

    try:
        response_content_json_str, usage_info = _call_gpt_with_retry(
            client,
            model_name,
            messages,
            max_completion_tokens,
            temperature,
            tools=[SECTION_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "extract_section_details"}}
        )

        parsed_data = parse_gpt_json_response(
            response_content_json_str,
            expected_keys=["section_summary", "section_tags", "section_standard", "section_standard_codes"]
        )

        if usage_info:
             prompt_tokens = usage_info.prompt_tokens
             completion_tokens = usage_info.completion_tokens
             total_tokens = usage_info.total_tokens
             prompt_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST
             completion_cost = (completion_tokens / 1000) * COMPLETION_TOKEN_COST
             total_cost = prompt_cost + completion_cost
             logging.info(f"API Usage (Section: {section_id}) - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Cost: ${total_cost:.4f}")
        else:
             logging.info(f"Usage information not available for section {section_id}.")

        return parsed_data

    except Exception as e:
        logging.error(f"Error processing section {section_id}: {e}", exc_info=True)
        return None

# --- Main Execution Logic ---

def main():
    """Main function to orchestrate section detail generation."""
    logging.info("Starting main execution...")
    SECTION_DETAILS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and group all chunk data by chapter
    all_chapters_data = load_all_chunk_data_grouped(CHUNK_INPUT_DIR)
    if not all_chapters_data:
        logging.error("Failed to load or group chunk data. Aborting.")
        return

    # 2. Create OpenAI client
    client = create_openai_client(BASE_URL)
    if not client:
        logging.error("Failed to create OpenAI client. Aborting.")
        return

    processed_sections = 0
    failed_sections = 0
    skipped_sections = 0
    processed_chapters = 0
    failed_chapters_loading_details = 0
    chapters_to_process_keys = []

    # 3. Determine which chapters to process based on PROCESS_CHAPTER
    if PROCESS_CHAPTER is None or str(PROCESS_CHAPTER).lower() == "all":
        chapters_to_process_keys = list(all_chapters_data.keys())
        logging.info(f"Processing all {len(chapters_to_process_keys)} chapters found.")
    else:
        # Convert PROCESS_CHAPTER to the type of the keys in all_chapters_data (could be int or str)
        target_chapter_key = None
        if all_chapters_data:
            first_key = next(iter(all_chapters_data))
            key_type = type(first_key)
            try:
                target_chapter_key = key_type(PROCESS_CHAPTER)
            except (ValueError, TypeError):
                 logging.error(f"Could not convert PROCESS_CHAPTER ('{PROCESS_CHAPTER}') to the key type ({key_type}).")
                 return

        if target_chapter_key in all_chapters_data:
            chapters_to_process_keys.append(target_chapter_key)
            logging.info(f"Processing only specified chapter: {PROCESS_CHAPTER}")
        else:
            logging.error(f"Specified chapter {PROCESS_CHAPTER} (as type {key_type}) not found in loaded data.")
            logging.info(f"Available chapters: {list(all_chapters_data.keys())}")
            return

    # 4. Iterate through selected chapters
    for chapter_num in chapters_to_process_keys:
        chapter_chunks = all_chapters_data[chapter_num]
        logging.info(f"\n===== Processing Chapter: {chapter_num} =====")

        # 5. Load chapter details for context
        chapter_details = load_chapter_details(chapter_num, CHAPTER_DETAILS_INPUT_DIR)
        if not chapter_details:
            logging.warning(f"Could not load chapter details for chapter {chapter_num}. Skipping sections in this chapter.")
            failed_chapters_loading_details += 1
            continue # Skip to next chapter if context is missing

        processed_chapters += 1
        recent_section_summaries = [] # Initialize sliding window for this chapter

        # 6. Group chunks by section and determine processing order
        sections_in_chapter_grouped = group_chunks_by_section(chapter_chunks)
        if not sections_in_chapter_grouped:
            logging.warning(f"No sections found or grouped for chapter {chapter_num}.")
            continue

        # Determine section order based on the 'order' of the first chunk in each section
        section_order_map = {}
        for section_id, chunks in sections_in_chapter_grouped.items():
            if chunks:
                section_order_map[section_id] = chunks[0].get('order', float('inf')) # Use order of first chunk
            else:
                section_order_map[section_id] = float('inf') # Should not happen if grouping is correct

        sorted_section_ids = sorted(sections_in_chapter_grouped.keys(), key=lambda sid: section_order_map[sid])
        logging.info(f"Found {len(sorted_section_ids)} sections in chapter {chapter_num}. Processing in order.")


        # 7. Iterate through sections in determined order
        for section_id in sorted_section_ids:
            section_chunks = sections_in_chapter_grouped[section_id]

            # Sanitize section_id for filename
            # Use a consistent way to generate filename, maybe hash long IDs?
            safe_section_id_str = str(section_id).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').strip()
            # Truncate if too long?
            max_len = 100
            safe_section_id = (safe_section_id_str[:max_len] + '...') if len(safe_section_id_str) > max_len else safe_section_id_str
            if not safe_section_id:
                safe_section_id = "unknown_section" # Fallback for empty IDs
            output_filename = f"chapter_{chapter_num}_section_{safe_section_id}_details.json"
            output_filepath = SECTION_DETAILS_OUTPUT_DIR / output_filename

            # 8. Check if output already exists (for resuming)
            if output_filepath.exists():
                logging.info(f"Section details already exist for '{section_id}' (File: {output_filename}). Skipping generation.")
                # OPTIONAL: Load the existing summary to add to context if needed for subsequent sections?
                # try:
                #     with open(output_filepath, 'r', encoding='utf-8') as f:
                #         existing_data = json.load(f)
                #     if 'section_summary' in existing_data:
                #          # Add to recent_section_summaries without exceeding limit
                #          recent_section_summaries.append(existing_data['section_summary'])
                #          if len(recent_section_summaries) > MAX_RECENT_SUMMARIES:
                #              recent_section_summaries.pop(0) # Remove oldest
                # except Exception as e:
                #     logging.warning(f"Could not load existing summary from {output_filename}: {e}")
                skipped_sections += 1
                continue

            # 9. Prepare context (sliding window of previous summaries)
            context_summaries = recent_section_summaries[-MAX_RECENT_SUMMARIES:] # Get last N summaries

            # 10. Process the section
            section_result = process_section(
                section_id=section_id,
                section_chunks=section_chunks,
                chapter_details=chapter_details,
                previous_section_summaries=context_summaries,
                client=client,
                model_name=MODEL_NAME,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=TEMPERATURE
            )

            # 11. Save results and update context window
            if section_result:
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(section_result, f, indent=4)
                    logging.info(f"Successfully saved section details for '{section_id}' to {output_filename}")
                    processed_sections += 1

                    # Add new summary to sliding window
                    new_summary = section_result.get('section_summary')
                    if new_summary:
                        recent_section_summaries.append(new_summary)
                        if len(recent_section_summaries) > MAX_RECENT_SUMMARIES:
                            recent_section_summaries.pop(0) # Remove oldest summary

                except Exception as e:
                    logging.error(f"Error saving section details for '{section_id}' to {output_filename}: {e}", exc_info=True)
                    failed_sections += 1
            else:
                logging.error(f"Failed to generate details for section '{section_id}'.")
                failed_sections += 1

    # --- Summary ---
    logging.info("\n===== Section-Level Detail Generation Summary =====")
    logging.info(f"Chapters targeted for processing: {chapters_to_process_keys}")
    logging.info(f"Chapters processed (details loaded): {processed_chapters}")
    logging.info(f"Chapters skipped (missing details): {failed_chapters_loading_details}")
    logging.info(f"Total sections processed/generated: {processed_sections}")
    logging.info(f"Total sections skipped (already existed): {skipped_sections}")
    logging.info(f"Total sections failed: {failed_sections}")
    logging.info("===================================================")
    logging.info("Section-Level Detail Generation Finished.")


if __name__ == "__main__":
    main()
