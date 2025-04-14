#!/usr/bin/env python3
"""
Utilities for generating chapter, section, and chunk details using OpenAI API
via RBC's custom endpoint.
"""

import os
import requests
import json
import time
import traceback
import natsort
from collections import defaultdict
from openai import OpenAI, APIError
from pathlib import Path
import tiktoken # Added dependency
# Removed: from pipeline_utils import count_tokens

# --- Inlined from pipeline_utils ---

# Initialize tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    tokenizer = None

def count_tokens(text):
    """Count tokens in text using the tokenizer."""
    if not text:
        return 0
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            return len(text) // 4 # Fallback
    else:
        return len(text) // 4

# --- End Inlined ---


# ============ CONFIGURATION CONSTANTS (Modify or load securely) ============
CHUNK_INPUT_DIR = "2E_final_merged_chunks" # Directory containing chunk JSONs
CHAPTER_DETAILS_OUTPUT_DIR = "3A_chapter_details" # Directory to save chapter details JSONs
BASE_URL = "https://api.example.com/v1"
MODEL_NAME = "gpt-4-turbo-nonp" # Or the specific model you intend to use
MAX_COMPLETION_TOKENS = 2000 # Max tokens for the *response*
TEMPERATURE = 0.5 # Adjust as needed for creativity vs consistency

# OAuth settings - **LOAD SECURELY (e.g., environment variables)**
OAUTH_URL = "https://api.example.com/oauth/token"
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

# ============ HELPER FUNCTIONS ============

def _setup_ssl(source_path=SSL_SOURCE_PATH, local_path=SSL_LOCAL_PATH):
    """Copies SSL cert locally and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True # Already configured

    print("Setting up SSL certificate...")
    try:
        source = Path(source_path)
        local = Path(local_path)

        if not source.is_file():
            print(f"Error: SSL source certificate not found at {source_path}")
            return False

        local.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as source_file:
            content = source_file.read()
        with open(local, "wb") as dest_file:
            dest_file.write(content)

        os.environ["SSL_CERT_FILE"] = str(local)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local)
        print(f"SSL certificate configured successfully at: {local}")
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        print(f"Error setting up SSL certificate: {e}")
        traceback.print_exc()
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH):
    """Retrieves OAuth token from the specified endpoint."""
    print("Attempting to get OAuth token...")
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
            print("Error: 'access_token' not found in OAuth response.")
            return None
        print("OAuth token obtained successfully.")
        return oauth_token
    except requests.exceptions.RequestException as e:
        print(f"Error getting OAuth token: {e}")
        traceback.print_exc()
        return None

def create_openai_client(base_url=BASE_URL):
    """Sets up SSL, gets OAuth token, and creates the OpenAI client."""
    if not _setup_ssl():
        return None # SSL setup failed

    api_key = _get_oauth_token()
    if not api_key:
        return None # Token retrieval failed

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("OpenAI client created successfully.")
        return client
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        traceback.print_exc()
        return None

def _call_gpt_with_retry(client, model, messages, max_tokens, temperature, tools=None, tool_choice=None):
    """Makes the API call with retry logic, supporting tool calls."""
    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            print(f"Making API call (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...")
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
                print("Making API call with tool choice...")
            else:
                print("Making API call without tool choice (using default JSON format)...")
                completion_kwargs["response_format"] = {"type": "json_object"}


            response = client.chat.completions.create(**completion_kwargs)

            print("API call successful.")

            response_message = response.choices[0].message

            if tools and tool_choice:
                if not response_message.tool_calls:
                    fallback_content = response_message.content or "No content"
                    print(f"Warning: Model did not return tool calls as expected. Fallback content: {fallback_content[:200]}...")
                    raise ValueError("Expected tool calls in response, but none found.")

                tool_call = response_message.tool_calls[0]
                expected_tool_name = None
                if isinstance(tool_choice, str) and tool_choice != "auto" and tool_choice != "required":
                     expected_tool_name = tool_choice
                elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                     expected_tool_name = tool_choice.get("function", {}).get("name")


                if not expected_tool_name or tool_call.function.name != expected_tool_name:
                     raise ValueError(f"Expected tool '{expected_tool_name}' but received '{tool_call.function.name}'")

                function_args_json = tool_call.function.arguments
                return function_args_json, response.usage
            else:
                if not response_message.content:
                    raise ValueError("Invalid response structure received from API (no content).")
                return response_message.content, response.usage

        except APIError as e:
            print(f"API Error on attempt {attempt + 1}: {e}")
            last_exception = e
            time.sleep(_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"Non-API Error on attempt {attempt + 1}: {e}")
            last_exception = e
            time.sleep(_RETRY_DELAY)

    print(f"API call failed after {_RETRY_ATTEMPTS} attempts.")
    if last_exception:
        raise last_exception
    else:
        raise Exception("API call failed for unknown reasons.")


def parse_gpt_json_response(response_content_str, expected_keys):
    """Parses JSON response from GPT and validates expected keys."""
    try:
        data = json.loads(response_content_str)
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object.")

        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing expected keys in response: {', '.join(missing_keys)}")

        if 'tags' in expected_keys and not isinstance(data.get('tags'), list):
             print("Warning: 'tags' field is not a list.")
        if 'summary' in expected_keys and not isinstance(data.get('summary'), str):
             print("Warning: 'summary' field is not a string.")

        print("GPT JSON response parsed successfully.")
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding GPT JSON response: {e}")
        print(f"Raw response string: {response_content_str[:500]}...")
        return None
    except ValueError as e:
        print(f"Error validating GPT JSON response: {e}")
        print(f"Raw response string: {response_content_str[:500]}...")
        return None

# ============ PHASE 1: CHAPTER LEVEL FUNCTIONS ============

def _build_chapter_prompt(segment_text, prev_summary=None, prev_tags=None, is_final_segment=False):
    """Builds the messages list for the chapter/segment processing call using CO-STAR and XML, with enhanced detail."""

    system_prompt = """<role>You are an expert financial reporting specialist with deep knowledge of IFRS and US GAAP.</role>
<source_material>You are analyzing segments of a chapter from a comprehensive EY technical accounting guidance manual.</source_material>
<task>Your primary task is to extract key information and generate a highly detailed, structured summary and a set of specific, granular topic tags for the provided text segment. This output will be used to build a knowledge base for accurate retrieval by accounting professionals. You will provide the output using the available 'extract_chapter_details' tool.</task>
<guardrails>Base your analysis strictly on the provided text segment and any previous context given. Do not infer information not explicitly present or heavily implied. Focus on factual extraction and objective summarization. Ensure tags are precise and directly relevant to accounting standards, concepts, or procedures mentioned.</guardrails>"""

    user_prompt_elements = ["<prompt>"]

    user_prompt_elements.append("<context>You are processing a text segment from a chapter within an EY technical accounting guidance manual (likely IFRS or US GAAP focused). The ultimate goal is to populate a knowledge base for efficient and accurate information retrieval by accounting professionals performing research.</context>")
    if prev_summary:
        user_prompt_elements.append(f"<previous_summary>{prev_summary}</previous_summary>")
    if prev_tags:
        user_prompt_elements.append(f"<previous_tags>{json.dumps(prev_tags)}</previous_tags>")

    user_prompt_elements.append("<style>Highly detailed, structured, technical, analytical, precise, and informative. Use clear headings within the summary string as specified.</style>")
    user_prompt_elements.append("<tone>Professional, objective, expert.</tone>")
    user_prompt_elements.append("<audience>Accounting professionals needing specific guidance; requires accuracy, completeness (within the scope of the text), and easy identification of key concepts.</audience>")
    user_prompt_elements.append('<response_format>Use the "extract_chapter_details" tool to provide the summary and tags.</response_format>')

    user_prompt_elements.append(f"<current_segment>{segment_text}</current_segment>")

    user_prompt_elements.append("<instructions>")
    summary_structure_guidance = """
    **Summary Structure Guidance:** Structure the 'summary' string using the following headings. Provide detailed information under each:
    *   **Purpose:** Concisely state the primary objective of this chapter/segment. What core accounting problem, transaction type, or reporting area does it address?
    *   **Key Topics/Standards:** List the primary IFRS/GAAP standards (e.g., IFRS 16, IAS 36, ASC 842, ASC 360) explicitly mentioned or clearly relevant. Detail the specific, significant topics, concepts, principles, or procedures discussed (e.g., lease classification criteria, measurement of right-of-use asset, disclosure requirements for financial instruments, impairment testing steps for goodwill, criteria for revenue recognition). Mention key definitions if central to understanding the guidance. Be specific (e.g., instead of 'measurement', specify 'initial measurement' or 'subsequent measurement').
    *   **Context/Applicability:** Describe the scope precisely. What types of entities, transactions, industries, assets/liabilities, or specific situations does this guidance apply to? Crucially, mention any significant exceptions, scope limitations, or practical expedients noted in the text.
    *   **Key Outcomes/Decisions:** Identify the main outcomes, critical judgments, accounting policy choices, or key decisions an accountant needs to make based on applying this guidance (e.g., determining the lease term, classifying a contract modification, assessing control over an asset, selecting an impairment model).
    """
    tag_guidance = """
    **Tag Generation Guidance:** Generate specific, granular tags highly relevant for retrieval by accounting professionals. Include:
    *   Relevant standard names and specific paragraph numbers if frequently cited (e.g., 'IFRS 15', 'ASC 606', 'IAS 36.12').
    *   Core accounting concepts discussed (e.g., 'revenue recognition', 'lease modification accounting', 'goodwill impairment', 'functional currency determination').
    *   Specific procedures or models mentioned (e.g., 'five-step revenue model', 'right-of-use asset measurement', 'expected credit loss model').
    *   Key terms defined or central to the topic (e.g., 'performance obligation', 'lease term', 'cash-generating unit', 'significant financing component').
    *   Applicability context if specific (e.g., 'SME considerations', 'interim reporting', 'specific industry guidance').
    Aim for 5-15 highly relevant and specific tags. Avoid overly generic tags.
    """

    if prev_summary or prev_tags:
        user_prompt_elements.append(summary_structure_guidance)
        user_prompt_elements.append(tag_guidance)
        if is_final_segment:
            user_prompt_elements.append("**Objective:** Consolidate all provided context (<previous_summary>, <previous_tags>) with the <current_segment> to generate the FINAL, comprehensive chapter summary and tag set.")
            user_prompt_elements.append("**Action:** Synthesize all information. Ensure the final summary reflects the entirety of the chapter's content processed so far, adhering strictly to the detailed Summary Structure Guidance. Ensure the final tags list is comprehensive and refined based on all segments. Preserve critical details from all segments.")
        else:
            user_prompt_elements.append("**Objective:** Refine the cumulative understanding by incorporating the <current_segment>.")
            user_prompt_elements.append("**Action:** Integrate the <current_segment> information with the <previous_summary> and <previous_tags>. Provide an UPDATED detailed summary and a refined list of tags, adhering strictly to the detailed Summary Structure and Tag Generation Guidance. Ensure no loss of critical details from previous context.")
    else:
        user_prompt_elements.append(summary_structure_guidance)
        user_prompt_elements.append(tag_guidance)
        user_prompt_elements.append("**Objective:** Analyze the initial segment of the chapter.")
        user_prompt_elements.append("**Action:** Generate a detailed summary and a list of relevant topic tags based ONLY on the <current_segment>, adhering strictly to the detailed Summary Structure and Tag Generation Guidance.")

    user_prompt_elements.append("</instructions>")
    user_prompt_elements.append("</prompt>")
    user_prompt = "\n".join(user_prompt_elements)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def process_chapter_segment(segment_text, client, model_name, max_completion_tokens, temperature, prev_summary=None, prev_tags=None, is_final_segment=False):
    """Processes a single chapter segment using GPT."""
    messages = _build_chapter_prompt(segment_text, prev_summary, prev_tags, is_final_segment)

    prompt_tokens_est = sum(count_tokens(msg["content"]) for msg in messages) # Uses inlined function
    print(f"Estimated prompt tokens for segment: {prompt_tokens_est}")

    chapter_details_tool = {
        "type": "function",
        "function": {
            "name": "extract_chapter_details",
            "description": "Extracts the summary and topic tags from a chapter segment based on provided guidance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A detailed summary of the chapter segment, following the structure outlined in the prompt (Purpose, Key Topics/Standards, Context/Applicability, Key Outcomes/Decisions)."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of 5-15 specific, granular topic tags relevant for retrieval by accounting professionals (e.g., standard names/paragraphs, core concepts, procedures, key terms, applicability)."
                    }
                },
                "required": ["summary", "tags"]
            }
        }
    }

    try:
        response_content_json_str, usage_info = _call_gpt_with_retry(
            client,
            model_name,
            messages,
            max_completion_tokens,
            temperature,
            tools=[chapter_details_tool],
            tool_choice={"type": "function", "function": {"name": "extract_chapter_details"}}
        )

        parsed_data = parse_gpt_json_response(response_content_json_str, expected_keys=["summary", "tags"])

        if usage_info:
             prompt_tokens = usage_info.prompt_tokens
             completion_tokens = usage_info.completion_tokens
             total_tokens = usage_info.total_tokens
             prompt_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST
             completion_cost = (completion_tokens / 1000) * COMPLETION_TOKEN_COST
             total_cost = prompt_cost + completion_cost
             print(f"API Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Cost: ${total_cost:.4f}")
        else:
             print("Usage information not available.")

        return parsed_data

    except Exception as e:
        print(f"Error processing chapter segment: {e}")
        traceback.print_exc()
        return None


def get_chapter_level_details(chapter_number, chapter_chunks_data, client, model_name=MODEL_NAME, input_token_limit=80000, max_completion_tokens=MAX_COMPLETION_TOKENS, temperature=TEMPERATURE):
    """
    Generates summary and tags for a full chapter, handling large chapters recursively.

    Args:
        chapter_number (int or str): Identifier for the chapter.
        chapter_chunks_data (list): List of chunk dicts for this chapter, sorted correctly.
                                    Each dict needs 'content' and 'chunk_token_count'.
        client (OpenAI): Initialized OpenAI client.
        model_name (str): Name of the model to use.
        input_token_limit (int): Approximate token limit for GPT input.
        max_completion_tokens (int): Max tokens for the response.
        temperature (float): Sampling temperature.

    Returns:
        dict: {'chapter_summary': str, 'chapter_tags': list} or None if processing fails.
    """
    print(f"\n--- Processing Chapter {chapter_number} ---")
    if not chapter_chunks_data:
        print("Warning: No chunks provided for this chapter.")
        return None

    full_chapter_text_parts = []
    total_tokens = 0
    current_section_id = None
    for chunk in chapter_chunks_data:
        section_id = chunk.get('section_hierarchy', chunk.get('section_title', 'Unknown Section'))
        if section_id != current_section_id:
             full_chapter_text_parts.append(f"\n\n## Section: {section_id}\n\n")
             current_section_id = section_id
        full_chapter_text_parts.append(chunk.get('content', ''))
        total_tokens += chunk.get('chunk_token_count', 0)

    full_chapter_text = "".join(full_chapter_text_parts).strip()
    print(f"Total estimated tokens for chapter {chapter_number}: {total_tokens}")

    os.makedirs(CHAPTER_DETAILS_OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(CHAPTER_DETAILS_OUTPUT_DIR, f"chapter_{chapter_number}_details.json")

    if os.path.exists(output_filepath):
        print(f"Chapter {chapter_number} details already exist at {output_filepath}. Skipping generation.")
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            return existing_data
        except Exception as e:
            print(f"Warning: Could not load existing file {output_filepath}: {e}. Will attempt regeneration.")

    processing_limit = input_token_limit - max_completion_tokens - 1000
    final_chapter_details = None

    if total_tokens <= processing_limit:
        print("Processing chapter in a single call.")
        result = process_chapter_segment(full_chapter_text, client, model_name, max_completion_tokens, temperature, is_final_segment=True)
        if result:
            final_chapter_details = {'chapter_summary': result.get('summary'), 'chapter_tags': result.get('tags')}
        else:
            print(f"Failed to process chapter {chapter_number} in single call.")
    else:
        print(f"Chapter {chapter_number} exceeds token limit ({total_tokens} > {processing_limit}). Processing in segments.")
        segments = []
        current_segment_text = []
        current_segment_tokens = 0

        for chunk in chapter_chunks_data:
            chunk_text = chunk.get('content', '')
            chunk_tokens = chunk.get('chunk_token_count', 0)

            if current_segment_tokens > 0 and current_segment_tokens + chunk_tokens > processing_limit:
                segments.append("".join(current_segment_text).strip())
                current_segment_text = [chunk_text]
                current_segment_tokens = chunk_tokens
            else:
                current_segment_text.append(chunk_text)
                current_segment_tokens += chunk_tokens

        if current_segment_text:
            segments.append("".join(current_segment_text).strip())

        print(f"Divided chapter into {len(segments)} segments.")

        current_summary = None
        current_tags = None
        final_result = None

        for i, segment_text in enumerate(segments):
            print(f"Processing segment {i + 1}/{len(segments)}...")
            is_final = (i == len(segments) - 1)
            segment_result = process_chapter_segment(
                segment_text,
                client,
                model_name,
                max_completion_tokens,
                temperature,
                prev_summary=current_summary,
                prev_tags=current_tags,
                is_final_segment=is_final
            )

            if segment_result:
                current_summary = segment_result.get('summary')
                current_tags = segment_result.get('tags')
                if is_final:
                    final_result = segment_result
                print(f"Segment {i + 1} processed.")
            else:
                print(f"Error processing segment {i + 1} for chapter {chapter_number}. Aborting chapter.")
                return None

        if final_result:
             print(f"Successfully processed all segments for chapter {chapter_number}.")
             final_chapter_details = {'chapter_summary': final_result.get('summary'), 'chapter_tags': final_result.get('tags')}
        else:
             print(f"Failed to get final result after processing segments for chapter {chapter_number}.")

    if final_chapter_details:
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_chapter_details, f, indent=4)
            print(f"Successfully saved chapter {chapter_number} details to {output_filepath}")
        except Exception as e:
            print(f"Error saving chapter {chapter_number} details to {output_filepath}: {e}")
            traceback.print_exc()

    return final_chapter_details


# ============ PHASE 2 & 3 FUNCTIONS (Placeholders) ============

# TODO: Implement get_section_level_details
# TODO: Implement get_chunk_level_details


# ============ DATA LOADING FOR TESTING ============

def load_all_chunk_data_grouped(input_dir=CHUNK_INPUT_DIR):
    """
    Loads all chunk JSONs, sorts them by the 'order' field within the JSON,
    and then groups them by chapter_number.
    """
    all_chunks_data = []
    print(f"Loading chunks from: {input_dir}")
    try:
        filenames = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        if not filenames:
            print(f"Warning: No JSON files found in {input_dir}")
            return None # Changed from returning empty dict to None for consistency
        print(f"Found {len(filenames)} chunk files. Loading data...")
    except FileNotFoundError:
        print(f"Error: Input directory not found: {input_dir}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred listing files: {e}")
        traceback.print_exc()
        return None

    loaded_count = 0
    error_count = 0
    skipped_missing_sequence = 0
    skipped_missing_chapter = 0

    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- Validation ---
            # Check for 'sequence_number' instead of 'order'
            if 'sequence_number' not in data or not isinstance(data['sequence_number'], int):
                print(f"Warning: Missing or invalid 'sequence_number' field in {filename}. Skipping.")
                skipped_missing_sequence += 1
                continue

            chapter_number = data.get('chapter_number')
            if chapter_number is None:
                # Use sequence_number in the warning message
                print(f"Warning: Missing 'chapter_number' in {filename} (Sequence: {data['sequence_number']}). Skipping.")
                skipped_missing_chapter += 1
                continue

            if 'content' not in data or 'chunk_token_count' not in data:
                 # Use sequence_number in the warning message
                 print(f"Warning: Missing 'content' or 'chunk_token_count' in {filename} (Sequence: {data['sequence_number']}). Skipping.")
                 error_count += 1
                 continue
            # --- End Validation ---

            all_chunks_data.append(data)
            loaded_count += 1

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}. Skipping.")
            error_count += 1
        except Exception as e:
            print(f"Error processing file {filename}: {e}. Skipping.")
            traceback.print_exc()
            error_count += 1

    print(f"Successfully loaded data for {loaded_count} chunks.")
    # Update warning message
    if skipped_missing_sequence > 0:
        print(f"Skipped {skipped_missing_sequence} chunks missing or invalid 'sequence_number'.")
    if skipped_missing_chapter > 0:
         print(f"Skipped {skipped_missing_chapter} chunks missing 'chapter_number'.")
    if error_count > 0:
        print(f"Skipped {error_count} chunks due to other errors.")

    if not all_chunks_data:
        print("No valid chunks were loaded.")
        return None

    # --- Sort by the 'sequence_number' field ---
    try:
        # Sort by 'sequence_number'
        all_chunks_data.sort(key=lambda x: x['sequence_number'])
        print(f"Successfully sorted {len(all_chunks_data)} chunks by 'sequence_number' field.")
    except KeyError:
        # This shouldn't happen due to validation above, but as a safeguard
        # Update error message
        print("Error: Sorting failed because 'sequence_number' key was missing in some loaded chunks.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during sorting: {e}")
        traceback.print_exc()
        return None

    # --- Group by chapter_number ---
    grouped_by_chapter = defaultdict(list)
    for chunk in all_chunks_data:
        grouped_by_chapter[chunk['chapter_number']].append(chunk)

    # --- Sort chapters naturally (optional but good practice) ---
    # Use natsort if available for chapter keys
    try:
        if natsort:
            sorted_grouped_by_chapter = dict(natsort.natsorted(grouped_by_chapter.items()))
            print(f"Grouped data into {len(sorted_grouped_by_chapter)} chapters (naturally sorted).")
        else:
            # Sort chapter keys using standard sort if natsort is unavailable
            sorted_grouped_by_chapter = dict(sorted(grouped_by_chapter.items()))
            print(f"Grouped data into {len(sorted_grouped_by_chapter)} chapters (standard sort).")
    except Exception as e:
        print(f"Warning: Could not sort chapter keys: {e}. Returning unsorted chapters.")
        sorted_grouped_by_chapter = dict(grouped_by_chapter) # Return unsorted if error

    return sorted_grouped_by_chapter


# ============ MAIN EXECUTION FOR TESTING ============

if __name__ == "__main__":
    print("Running Chapter-Level Detail Generation...")

    PROCESS_CHAPTER = None
    TEST_INPUT_TOKEN_LIMIT = 80000

    all_chapters_data = load_all_chunk_data_grouped()

    if not all_chapters_data:
        print("Failed to load chunk data. Aborting test.")
        exit()

    chapters_to_process = []
    if PROCESS_CHAPTER is None or str(PROCESS_CHAPTER).lower() == "all":
        chapters_to_process = list(all_chapters_data.keys())
        print(f"Processing all {len(chapters_to_process)} chapters found.")
    else:
        test_chapter_str = str(PROCESS_CHAPTER)
        test_chapter_int = None
        try:
            test_chapter_int = int(PROCESS_CHAPTER)
        except ValueError:
            pass

        if test_chapter_str in all_chapters_data:
            chapters_to_process.append(test_chapter_str)
        elif test_chapter_int is not None and test_chapter_int in all_chapters_data:
             chapters_to_process.append(test_chapter_int)

        if not chapters_to_process:
            print(f"Error: No data found for specified chapter: {PROCESS_CHAPTER}")
            print(f"Available chapters: {list(all_chapters_data.keys())}")
            exit()
        else:
            print(f"Processing only specified chapter: {PROCESS_CHAPTER}")

    client = create_openai_client()

    if not client:
        print("Failed to create OpenAI client. Aborting test.")
        exit()

    processed_count = 0
    failed_count = 0
    skipped_count = 0

    for chapter_num in chapters_to_process:
        print(f"\n===== Processing Chapter: {chapter_num} =====")
        chapter_data = all_chapters_data.get(chapter_num)

        if not chapter_data:
             print(f"Error: Could not retrieve chunk data for chapter {chapter_num} from loaded data. Skipping.")
             failed_count += 1
             continue

        output_filepath = os.path.join(CHAPTER_DETAILS_OUTPUT_DIR, f"chapter_{chapter_num}_details.json")
        if os.path.exists(output_filepath):
             print(f"Output file already exists: {output_filepath}. Skipping generation.")
             skipped_count += 1
             continue


        chapter_details = get_chapter_level_details(
            chapter_number=chapter_num,
            chapter_chunks_data=chapter_data,
            client=client,
            model_name=MODEL_NAME,
            input_token_limit=TEST_INPUT_TOKEN_LIMIT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=TEMPERATURE
        )

        if chapter_details:
            print(f"--- Successfully processed or loaded details for Chapter {chapter_num} ---")
            processed_count += 1
        else:
            print(f"--- Failed to generate or load details for Chapter {chapter_num} ---")
            failed_count += 1

    print("\n===== Chapter-Level Detail Generation Summary =====")
    print(f"Total chapters attempted: {len(chapters_to_process)}")
    print(f"Successfully processed/generated: {processed_count}")
    print(f"Skipped (already existed): {skipped_count}")
    print(f"Failed: {failed_count}")
    print("===================================================")

    print("\nChapter-Level Detail Generation Finished.")
