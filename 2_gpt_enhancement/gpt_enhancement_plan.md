Phase 1: Chapter-Level Analysis (Corresponds to `chapter_analysis_utils.py`)

Goal: To generate a high-level summary and relevant tags for each chapter, providing context for subsequent stages. This also handles the challenge of processing chapters that exceed the LLM's input token limit.
Input: All chunk JSON files from the `CHUNK_INPUT_DIR` (default: `2E_final_merged_chunks`). Each chunk JSON must contain `content`, `chapter_number`, and `chunk_token_count`.
Process:
Load & Group: Load all chunk JSONs using `load_all_chunk_data_grouped` (from `chapter_analysis_utils.py`), grouping them by `chapter_number` and sorting naturally. Skips chunks missing required fields.
Reconstruct & Check Size: For each chapter, reconstruct the full chapter text using `get_chapter_level_details`. This involves:
    *   Concatenating the `content` of its chunks.
    *   Adding `## Section: {section_id}` markers between different sections for better LLM context.
    *   Calculating the `total_tokens` by summing the pre-calculated `chunk_token_count` for all chunks in the chapter.
    *   Determining if `total_tokens` exceeds the `processing_limit` (calculated as `input_token_limit` - `max_completion_tokens` - buffer (e.g., 1000 tokens)).
Process Chapter:
If within limit (`total_tokens <= processing_limit`): Send the entire reconstructed chapter text to the LLM in a single API call via `process_chapter_segment` (with `is_final_segment=True`). The prompt (`_build_chapter_prompt`) uses a detailed CO-STAR/XML structure asking for a summary (Purpose, Key Topics/Standards, Context/Applicability, Key Outcomes/Decisions) and 5-15 granular tags, enforced via a tool call.
If exceeds limit (`total_tokens > processing_limit`): Implement the iterative segmentation approach within `get_chapter_level_details`:
    *   Divide the chapter's chunks into segments, each aiming to stay below the `processing_limit`.
    *   Process the first segment using the detailed prompt via `process_chapter_segment`.
    *   For each subsequent segment, call `process_chapter_segment` again, sending its text along with the `summary` and `tags` generated from the *previous* segment (passed as `prev_summary` and `prev_tags`, used within `<previous_summary>` and `<previous_tags>` XML tags in the prompt). The prompt instructs the LLM to refine/consolidate.
    *   The final segment is processed with `is_final_segment=True` to generate the consolidated chapter summary and tags.
API Interaction:
    *   Uses `create_openai_client` which handles SSL setup (`_setup_ssl`) and OAuth token retrieval (`_get_oauth_token`) using environment variables/configuration for credentials and certificate paths.
    *   Uses `_call_gpt_with_retry` for API calls, incorporating retry logic.
    *   Crucially, uses OpenAI's **tool calling** feature. A specific tool schema (`extract_chapter_details`) is defined with `summary` (string) and `tags` (array of strings) parameters. The API call forces the use of this tool (`tool_choice={"type": "function", "function": {"name": "extract_chapter_details"}}`).
    *   The JSON arguments string returned by the tool call is parsed using `parse_gpt_json_response` to validate the structure.
Output: The final dictionary `{'chapter_summary': str, 'chapter_tags': list}` is **saved** as a JSON file for each chapter in the `CHAPTER_DETAILS_OUTPUT_DIR` (default: `3A_chapter_details`), named `chapter_{chapter_number}_details.json`. The function also returns this dictionary. Checks for existing files to allow skipping regeneration.

Phase 2: Section-Level Analysis

Goal: To generate more specific details for each section within a chapter, leveraging the chapter-level context obtained in Phase 6.1.
Input:
Chunks grouped by section identifier (e.g., section_hierarchy or section_title) within each chapter.
The chapter_summary and chapter_tags generated in Phase 6.1 for the corresponding chapter.
Process:
Iterate Sections: Loop through each section within each chapter.
Reconstruct Section: Concatenate the content of all chunks belonging to the current section.
Build Prompt: Create a new detailed prompt (similar structure to Phase 6.1 but focused on section-level details). This prompt will include:
The reconstructed section text.
The chapter_summary and chapter_tags as context (e.g., within <chapter_context> tags).
Instructions to generate:
section_summary: Detailed summary specific to this section (following the Purpose, Topics, Context, Outcomes structure).
section_tags: Tags specific to this section.
section_standard: The primary standard applicable (e.g., "IFRS", "US GAAP", "N/A").
section_standard_codes: A list of specific standard codes mentioned (e.g., ["IFRS 16", "IAS 17"]).
API Call: Call the LLM using the constructed prompt.
Parse Response: Extract the generated section details.
Output: A temporary data structure mapping each section identifier (e.g., chapter + section hierarchy) to its generated section_summary, section_tags, section_standard, and section_standard_codes.
Phase 6.3: Chunk-Level Analysis & Final Assembly

Goal: To generate the final chunk-specific details (summary, importance score, references) using the section context, and assemble the complete chunk JSON for the database.
Input:
Individual chunk JSON files from 2E_final_merged_chunks.
The corresponding section-level details (section_summary, section_tags, section_standard, section_standard_codes) generated in Phase 6.2.
Process:
Iterate Chunks: Loop through each chunk JSON file.
Build Prompt: Create a prompt for the LLM including:
The chunk's content.
The relevant section_summary, section_tags, section_standard, and section_standard_codes as context (e.g., within <section_context> tags).
Instructions to generate:
summary: A concise summary of this specific chunk, explaining how it fits within the section's context.
importance_score: A float (0.0-1.0) indicating the chunk's importance relative to the overall section content.
section_references: A list of explicit references to other sections/paragraphs found within this chunk's text (e.g., ["See Section 4.5", "IAS 36.12"]). (This assumes we proceed with extracting references at the chunk level).
API Call: Call the LLM.
Parse Response: Extract the generated chunk summary, importance_score, and section_references.
Assemble Final Chunk:
Take the original chunk data.
Add the newly generated summary, importance_score, and section_references.
Add the tags, standard, and standard_codes based on the strategy we still need to confirm (e.g., inherit directly from the section-level results from Phase 6.2).
Ensure all other required schema fields (like id, document_id, chapter_name, embedding, text_search_vector, sequence_number, page_start, page_end, section_hierarchy, section_title, content) are present (some might need generation/calculation outside the LLM calls, e.g., embeddings, sequence numbers).
Save Output: Save the fully populated chunk JSON to a new final output directory (e.g., 2F_gpt_enhanced_chunks).
Output: The final directory (2F_gpt_enhanced_chunks) containing JSON files for each chunk, fully populated according to your schema and ready for database ingestion.
