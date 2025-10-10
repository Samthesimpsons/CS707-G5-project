"""Resume batch job processing for TVQA data.

This script handles Stage 2 of the batch processing pipeline - retrieving
completed batch job results and creating the output file.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from openai import OpenAI  # type: ignore[import-not-found]
from dotenv import load_dotenv  # type: ignore[import-not-found]


def load_batch_state(state_file: Path) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Load batch job state from file.

    Args:
        state_file: Path to the state file.

    Returns:
        Tuple of (batch_job_id, metadata).
    """
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")

    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    batch_job_id = state_data["batch_job_id"]
    metadata = state_data["metadata"]

    print(f"Loaded batch job state from: {state_file}")
    print(f"Batch job ID: {batch_job_id}\n")

    return batch_job_id, metadata


def check_batch_status(client: OpenAI, batch_job_id: str) -> Tuple[str, bool]:
    """Check batch job status without waiting.

    Args:
        client: OpenAI client instance.
        batch_job_id: Batch job ID.

    Returns:
        Tuple of (status, is_complete) where is_complete is True if job finished successfully.
    """
    batch_job = client.batches.retrieve(batch_job_id)
    status = batch_job.status

    is_complete = True if status == "completed" else False

    return status, is_complete


def retrieve_batch_results(client: OpenAI, batch_job_id: str) -> Dict[str, Any]:
    """Retrieve completed batch job and return results.

    Args:
        client: OpenAI client instance.
        batch_job_id: Batch job ID.

    Returns:
        Dictionary mapping custom_id to parsed event description JSON.
    """
    batch_job = client.batches.retrieve(batch_job_id)
    result_file_id = batch_job.output_file_id
    result_content = client.files.content(result_file_id).content

    results = {}
    for line in result_content.decode("utf-8").strip().split("\n"):
        result_obj = json.loads(line)
        custom_id = result_obj["custom_id"]

        content_str = result_obj["response"]["body"]["choices"][0]["message"]["content"]
        event_description_json = json.loads(content_str)

        results[custom_id] = event_description_json

    print(f"Retrieved {len(results)} results\n")
    return results


def create_output_from_results(
    metadata: Dict[str, Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    output_file: Path,
) -> None:
    """Create output file from batch results.

    Args:
        metadata: Metadata dictionary with clip information.
        results: Dictionary mapping custom_id to structured event description JSON.
        output_file: Path to save the output JSON file.
    """
    output_data = []

    for custom_id, clip_meta in metadata.items():
        event_data = results.get(custom_id)

        if event_data:
            output_entry = {
                "vid_name": clip_meta["vid_name"],
                "subjects": clip_meta["subjects"],
                "time": {
                    "start": clip_meta["start_time"],
                    "end": clip_meta["end_time"],
                    "duration_seconds": round(clip_meta["duration_seconds"], 2),
                },
                "event": {
                    "location": event_data.get("location", clip_meta["location"]),
                    "detailed_description": event_data.get("detailed_description", ""),
                    "summary": event_data.get("summary_verb_noun", ""),
                    "flashback_events": event_data.get("flashback_events", []),
                },
            }
        else:
            output_entry = {
                "vid_name": clip_meta["vid_name"],
                "subjects": clip_meta["subjects"],
                "time": {
                    "start": clip_meta["start_time"],
                    "end": clip_meta["end_time"],
                    "duration_seconds": round(clip_meta["duration_seconds"], 2),
                },
                "event": {
                    "location": clip_meta["location"],
                    "detailed_description": f"Scene at {clip_meta['location']} with {', '.join(clip_meta['subjects'][:3])}",
                    "summary": "",
                    "flashback_events": [],
                },
            }

        output_data.append(output_entry)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved output to: {output_file}")
    print(f"Total entries: {len(output_data)}\n")


def resume_tvqa_batch(client: OpenAI, state_file: Path, output_file: Path) -> None:
    """Process TVQA data - Stage 2: Resume from batch job completion.

    Args:
        client: OpenAI client instance.
        state_file: Path to the saved batch state file.
        output_file: Path to save the output JSON file.
    """
    print(f"\n{'='*80}")
    print("TVQA Batch Processing using OpenAI Batch API - Stage 2")
    print(f"{'='*80}\n")

    batch_job_id, metadata = load_batch_state(state_file)

    status, is_complete = check_batch_status(client, batch_job_id)

    if not is_complete:
        print(f"Batch job status: {status}")
        print("Job is still not complete. Please wait and try again later.")
        print(
            "\nNote: Batch jobs can take up to 24 hours but usually complete much faster."
        )
        print("You can check status at: https://platform.openai.com/batches")
        return

    results = retrieve_batch_results(client, batch_job_id)

    create_output_from_results(metadata, results, output_file)

    print("STAGE 2 COMPLETE:")
    print(f"Total clips processed: {len(metadata)}")
    print(f"Output file: {output_file}")


def processing_pipeline() -> None:
    """Main function to resume TVQA batch job processing."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    state_file = Path("./data/temp/batch_state_tvqa.json")
    output_file = Path("./data/tvqa_processed_output.json")

    try:
        resume_tvqa_batch(client, state_file, output_file)
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
