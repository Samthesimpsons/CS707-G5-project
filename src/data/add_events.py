"""
Event Extraction and Data Enhancement Script - Batch API Version

This script creates batch requests for event extraction from episode JSON files.
Instead of processing immediately, it submits all requests to OpenAI's Batch API
and returns a tracking link to monitor the batch job progress.

Note: The downloading and parsing of batch outputs occurs in generate_qa.py after
the batch job completes. This script only handles batch submission.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI
from textwrap import dedent
from tqdm import tqdm
from pydantic import BaseModel


class ExtractedEvent(BaseModel):
    """Pydantic model for event extraction from LLM."""

    event_description: str
    involved_subjects: List[str]
    start_offset: str
    end_offset: str


class EventsList(BaseModel):
    """Pydantic model for list of events returned by LLM."""

    events: List[ExtractedEvent]


@dataclass
class SceneRequest:
    """Represents metadata for a batch request."""

    custom_id: str
    json_file: str
    episode_id: str
    scene_idx: int


class BatchRequestBuilder:
    """Builds batch requests for OpenAI Batch API."""

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Initialize the batch request builder.

        Args:
            model: Model name to use (e.g., "gpt-4o", "gpt-4-turbo")
        """
        self.model = model
        print(f"Initialized Batch Request Builder with model: {model}")

    def _create_extraction_prompt(self, scene: Dict[str, Any], max_events: int) -> str:
        """
        Create prompt for event extraction.

        Args:
            scene: Scene dictionary containing dialogue, location, and other metadata
            max_events: Maximum number of events to extract

        Returns:
            Formatted prompt string for the LLM
        """
        dialogue_text = "\n".join(
            [f"{d['speaker']}: {d['text']}" for d in scene.get("dialogue", [])]
        )

        prompt = dedent(
            f"""You are an expert at analyzing narrative scenes and extracting atomic events.

            Scene Information:
            Location: {scene.get('location', 'Unknown')}
            Characters Present: {', '.join(scene.get('subjects', []))}
            Duration: {scene.get('clip_start', '')} to {scene.get('clip_end', '')}
            Scene Description: {scene.get('scene_description', 'N/A')}

            Dialogue:
            {dialogue_text}

            Task: Extract up to {max_events} atomic events from this scene. Each event should be:
            1. A discrete, verifiable action or occurrence
            2. Clearly describe what happened (not vague like "they talked")
            3. Include only characters who actively participated (not just present)
            4. Have a specific temporal boundary within the scene
            5. Be plot-relevant (ignore trivial actions like "someone sat down")

            Guidelines:
            - Focus on significant actions, revelations, decisions, or emotional moments
            - Be specific: instead of "they argued", say "Ross and Susan argued about baby names"
            - Don't extract events from scene descriptions, only from actual dialogue/actions
            - Involved subjects should be all characters who were present in the environment, including those who were not speaking but were part of the scene.
            - If a character is mentioned in the event description but was not present in the scene, exclude them from the involved subjects
            - If a character showed up in the scene but left before the event occured, exclude them from the involved subjects

            Format your response as a JSON object with an "events" array:
            {{
                "events": [
                    {{
                        "event_description": "Clear, specific description of what happened",
                        "involved_subjects": ["CHARACTER1", "CHARACTER2"],
                        "start_offset": "00:00:05",
                        "end_offset": "00:00:20"
                    }},
                    {{
                        "event_description": "Another event description",
                        "involved_subjects": ["CHARACTER1"],
                        "start_offset": "00:00:21",
                        "end_offset": "00:00:35"
                    }}
                ]
            }}

            Output only the JSON object, no additional text."""
        )

        return prompt

    def create_batch_request(
        self,
        scene: Dict[str, Any],
        custom_id: str,
        max_events: int = 10,
    ) -> Dict[str, Any]:
        """
        Create a batch request object for a single scene.

        Converts the Pydantic EventsList model to a JSON schema for structured output
        compatibility with OpenAI's Batch API.

        Args:
            scene: Scene dictionary with dialogue, location, subjects, etc.
            custom_id: Unique identifier for this request
            max_events: Maximum number of events to extract (default: 10)

        Returns:
            Dictionary in OpenAI Batch API request format
        """
        prompt = self._create_extraction_prompt(scene, max_events)

        json_schema = {
            "name": "events_list",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "event_description": {"type": "string"},
                                "involved_subjects": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "start_offset": {"type": "string"},
                                "end_offset": {"type": "string"},
                            },
                            "required": [
                                "event_description",
                                "involved_subjects",
                                "start_offset",
                                "end_offset",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["events"],
                "additionalProperties": False,
            },
        }

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": json_schema},
            },
        }


def load_episode_data(json_path: Path) -> Dict[str, Any]:
    """
    Load episode data from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Dictionary containing episode data
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, output_path: Path, indent: int = 2) -> None:
    """
    Save data to JSON file with pretty formatting.

    Args:
        data: Data to save (must be JSON serializable)
        output_path: Path where to save the JSON file
        indent: Number of spaces for indentation (default: 2)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_jsonl(data: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save data to JSONL file (one JSON object per line).

    Args:
        data: List of dictionaries to save
        output_path: Path where to save the JSONL file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_batch_requests_for_episodes(
    json_files: List[Path],
    builder: BatchRequestBuilder,
    max_events_per_scene: int = 10,
) -> tuple[List[Dict[str, Any]], List[SceneRequest]]:
    """
    Create batch requests for all scenes in all episodes.

    Args:
        json_files: List of episode JSON file paths
        builder: BatchRequestBuilder instance
        max_events_per_scene: Maximum events to extract per scene

    Returns:
        Tuple of (batch_requests, metadata) where:
        - batch_requests: List of batch request dictionaries
        - metadata: List of SceneRequest objects with metadata for each request
    """
    batch_requests = []
    metadata = []

    for json_file in tqdm(json_files, desc="Creating batch requests"):
        try:
            episode_data = load_episode_data(json_file)
            episode_id = episode_data.get("episode", "unknown")
            scenes = episode_data.get("scenes", [])

            for scene_idx, scene in enumerate(scenes):
                custom_id = f"{json_file.stem}_scene_{scene_idx:04d}"

                batch_request = builder.create_batch_request(
                    scene, custom_id, max_events=max_events_per_scene
                )
                batch_requests.append(batch_request)

                scene_metadata = SceneRequest(
                    custom_id=custom_id,
                    json_file=str(json_file),
                    episode_id=episode_id,
                    scene_idx=scene_idx,
                )
                metadata.append(scene_metadata)

        except Exception as e:
            print(f"Warning: Error processing {json_file.name}: {e}")
            continue

    return batch_requests, metadata


def submit_batch_to_openai(
    batch_file_path: Path,
    description: str = "Event extraction batch",
) -> Dict[str, Any]:
    """
    Upload batch file and create batch job with OpenAI.

    First uploads the JSONL batch file to OpenAI's file storage, then creates
    a batch job that will process all requests within a 24-hour completion window.

    Args:
        batch_file_path: Path to the JSONL batch file
        description: Description for the batch job

    Returns:
        Dictionary with batch information including batch_id and status URL
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"\nUploading batch file: {batch_file_path}")
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    print(f"File uploaded successfully. File ID: {batch_input_file.id}")

    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )

    print("Batch job created successfully!")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": batch_input_file.id,
        "created_at": batch.created_at,
        "metadata": batch.metadata,
    }


def generate_events_from_subtitles(
    input_dir: Path,
    output_dir: Path,
    model: str = "gpt-4o",
    max_events_per_scene: int = 10,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create and submit batch requests for event extraction from all episodes.

    Args:
        input_dir: Input directory path containing JSON files
        output_dir: Output directory path for batch files and metadata
        model: Model name to use (e.g., "gpt-4o", "gpt-4-turbo")
        max_events_per_scene: Maximum events to extract per scene (default: 10)
        max_files: Maximum number of files to process, None for all (default: None)

    Returns:
        Dictionary containing batch information:
            - batch_id: OpenAI batch job ID
            - status: Current batch status
            - tracking_url: URL to track batch progress
            - total_requests: Total number of requests in the batch
            - batch_file: Path to the JSONL batch file
            - metadata_file: Path to the metadata JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))
    if max_files:
        json_files = json_files[:max_files]

    print(f"\n{'='*70}")
    print("BATCH REQUEST CREATION")
    print(f"{'='*70}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model:            {model}")
    print(f"Episodes:         {len(json_files)}")
    print(f"{'='*70}\n")

    builder = BatchRequestBuilder(model=model)
    batch_requests, metadata = create_batch_requests_for_episodes(
        json_files, builder, max_events_per_scene
    )

    print(
        f"\nCreated {len(batch_requests)} batch requests for {len(json_files)} episodes"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file_path = output_dir / f"batch_requests_{timestamp}.jsonl"
    save_jsonl(batch_requests, batch_file_path)
    print(f"Saved batch requests to: {batch_file_path}")

    metadata_file_path = output_dir / f"batch_metadata_{timestamp}.json"
    metadata_dict = {
        "timestamp": timestamp,
        "model": model,
        "max_events_per_scene": max_events_per_scene,
        "total_requests": len(batch_requests),
        "total_episodes": len(json_files),
        "requests": [
            {
                "custom_id": m.custom_id,
                "json_file": m.json_file,
                "episode_id": m.episode_id,
                "scene_idx": m.scene_idx,
            }
            for m in metadata
        ],
    }
    save_json(metadata_dict, metadata_file_path)
    print(f"Saved metadata to: {metadata_file_path}")

    print(f"\n{'='*70}")
    print("SUBMITTING BATCH TO OPENAI")
    print(f"{'='*70}")

    batch_info = submit_batch_to_openai(
        batch_file_path,
        description=f"Event extraction for {len(json_files)} episodes ({timestamp})",
    )

    batch_info["tracking_url"] = (
        f"https://platform.openai.com/batches/{batch_info['batch_id']}"
    )
    batch_info["total_requests"] = len(batch_requests)
    batch_info["batch_file"] = str(batch_file_path)
    batch_info["metadata_file"] = str(metadata_file_path)

    batch_info_path = output_dir / f"batch_info_{timestamp}.json"
    save_json(batch_info, batch_info_path)
    print(f"\nSaved batch info to: {batch_info_path}")

    print(f"\n{'='*70}")
    print("BATCH SUBMISSION COMPLETE")
    print(f"{'='*70}")
    print(f"Batch ID:         {batch_info['batch_id']}")
    print(f"Status:           {batch_info['status']}")
    print(f"Total Requests:   {batch_info['total_requests']}")
    print(f"Tracking URL:     {batch_info['tracking_url']}")
    print("\nTo check batch status, visit:")
    print(f"  {batch_info['tracking_url']}")
    print("\nOr use the OpenAI CLI:")
    print(f"  openai api batches.retrieve -i {batch_info['batch_id']}")
    print(f"{'='*70}\n")

    return batch_info
