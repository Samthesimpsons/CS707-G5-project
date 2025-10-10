"""
TODO: Add removal of videos at the end once split
TODO: Moment indexing naively will be t Time
TODO: Add one more which is a more detailed description event and another is a summarized one
TODO: Verb + Noun; Action-based summary; What is happening
TODO: Check bug fix for the timestamp and clip chunking
TODO: Flashbacks: []
"""

import json
import os
import re
import subprocess
from textwrap import dedent
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm  # type: ignore[import-not-found]
from openai import OpenAI  # type: ignore[import-not-found]
from dotenv import load_dotenv  # type: ignore[import-not-found]


LLM_MODEL = "gpt-4o-mini"


def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT timestamp to seconds.

    Args:
        time_str: Timestamp in format "HH:MM:SS,mmm" or "HH:MM:SS.mmm".

    Returns:
        Time in seconds as float.
    """
    time_str = time_str.replace(",", ".")
    match = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", time_str)
    if match:
        hours, minutes, seconds, milliseconds = match.groups()
        total_seconds = (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(milliseconds) / 1000.0
        )
        return total_seconds
    raise ValueError(f"Invalid SRT timestamp format: {time_str}")


def seconds_to_ffmpeg_time(seconds: float) -> str:
    """Convert seconds to ffmpeg timestamp format.

    Args:
        seconds: Time in seconds.

    Returns:
        Timestamp in format "HH:MM:SS.mmm".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def extract_unique_locations(synced_dir: Path) -> List[str]:
    """Extract all unique locations from synced JSON files.

    Args:
        synced_dir: Directory containing synced JSON files.

    Returns:
        List of unique location strings (raw from scene descriptions).
    """
    all_locations = set()

    synced_files = sorted(synced_dir.glob("*.json"))
    print(f"Extracting locations from {len(synced_files)} episodes...")

    for json_file in synced_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            scenes = data.get("scenes", [])
            for scene in scenes:
                scene_description = scene.get("scene_description", "").strip()
                if scene_description and scene_description != "Episode dialogue":
                    scene_description = re.sub(r"[^a-zA-Z0-9\s]", "", scene_description)
                    all_locations.add(scene_description)

        except Exception as e:
            print(f"Warning: Error processing {json_file.name}: {e}")

    unique_locations = sorted(list(all_locations))
    print(f"Found {len(unique_locations)} unique locations\n")

    return unique_locations


def generate_location_mappings_with_llm(unique_locations: List[str]) -> Dict[str, str]:
    """Use LLM to categorize locations into cleaner common locations.

    Args:
        unique_locations: List of raw location strings.

    Returns:
        Dictionary mapping raw location -> common location.
    """
    print(f"Categorizing {len(unique_locations)} locations in a single LLM call...")

    llm_prompt = dedent(
        f"""You are provided a list of scenes from the TV show, Friends.
        Your task: Create a mapping table of scene to location.

        LIST OF SCENES:
        {unique_locations}

        Return ONLY a JSON object mapping each scene to a common location. Format:
        {{
        "scene_1": "location_1",
        "scene_2": "location_2",
        "scene_3": "location_1",
        ...
        }}

        IMPORTANT:
        - Return ONLY the JSON object, no other text
        - Do NOT modify the original scene naming when using them as the keys for the mapping
        - Use common location names if the scenes are describing the same location
        """
    )

    try:
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are to strictly return only valid JSON.",
                },
                {"role": "user", "content": llm_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content.strip()
        mappings = json.loads(response_text)

        missing = set(unique_locations) - set(mappings.keys())
        if missing:
            print(f"Warning: {len(missing)} locations not mapped by LLM")
            for loc in missing:
                mappings[loc] = "Other"

        print(f"Successfully mapped {len(mappings)} locations\n")
        return mappings

    except Exception as e:
        print(f"Error in producing locations mapping: {e}, using fallback mappings\n")
        return {loc: "Other" for loc in unique_locations}


def prepare_batch_requests(
    synced_dir: Path,
    location_mappings: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Prepare batch requests for all event descriptions.

    Args:
        synced_dir: Directory containing synced JSON files.
        location_mappings: Dictionary mapping raw scene descriptions to common locations.

    Returns:
        Tuple of (batch_requests_list, metadata_dict) where metadata contains
        information needed to reconstruct memory tuples later.
    """
    batch_requests = []
    metadata = {}

    synced_files = sorted(synced_dir.glob("*.json"))
    print(f"Preparing batch requests for {len(synced_files)} episodes...")
    for synced_file in synced_files:
        try:
            with open(synced_file, "r", encoding="utf-8") as f:
                synced_data = json.load(f)

            episode_id = synced_data.get("episode", "unknown")
            scenes = synced_data.get("scenes", [])

            for scene_idx, scene in enumerate(scenes):
                scene_description = re.sub(
                    r"[^a-zA-Z0-9\s]", "", scene.get("scene_description", "Unknown")
                )
                dialogue = scene.get("dialogue", [])
                location = location_mappings.get(scene_description, "Other")
                subjects = list(
                    set(
                        [
                            entry.get("speaker", "UNKNOWN")
                            for entry in dialogue
                            if entry.get("speaker")
                        ]
                    )
                )
                subjects.sort()

                dialogue_text = "\n".join(
                    f"{entry.get('speaker', 'UNKNOWN')}: {entry.get('text', '').strip()}"
                    for entry in dialogue
                    if entry.get("text")
                )

                custom_id = f"{episode_id}-scene{scene_idx + 1:02d}"

                llm_prompt = dedent(
                    f"""Given the following scene from a TV show episode, summarise the scene into a brief event description of 1 sentence.

                        Examples of brief event description:
                        - Ross and Chandler are celeberating their birthday in Ross's Apartment
                        - Celeberate Ross's birthday
                        - 

                        Scene Marker: {scene_description}
                        Location: {location}
                        Characters Present: {', '.join(subjects)}

                        Dialogue:
                        {dialogue_text}

                        Event description: 1 sentence:"""
                )

                batch_request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": LLM_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that describes TV show scenes concisely.",
                            },
                            {"role": "user", "content": llm_prompt},
                        ],
                        "temperature": 0,
                    },
                }

                batch_requests.append(batch_request)

                times = [
                    (entry.get("start_time"), entry.get("end_time"))
                    for entry in dialogue
                    if entry.get("start_time") and entry.get("end_time")
                ]

                if times:
                    start_time = min([t[0] for t in times])
                    end_time = max([t[1] for t in times])
                    try:
                        duration_seconds = srt_time_to_seconds(
                            end_time
                        ) - srt_time_to_seconds(start_time)
                    except ValueError:
                        duration_seconds = 0.0
                else:
                    start_time = None
                    end_time = None
                    duration_seconds = 0.0

                metadata[custom_id] = {
                    "episode_id": episode_id,
                    "scene_idx": scene_idx,
                    "location": location,
                    "subjects": subjects,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration_seconds,
                }

        except Exception as e:
            print(f"Warning: Error processing {synced_file.name}: {e}")
            continue

    print(f"Prepared {len(batch_requests)} batch requests\n")
    return batch_requests, metadata


def submit_batch_job(
    client: OpenAI, batch_requests: List[Dict[str, Any]], temp_dir: Path
) -> str:
    """Submit batch job to OpenAI.

    Args:
        client: OpenAI client instance.
        batch_requests: List of batch request dictionaries.
        temp_dir: Directory to store temporary JSONL file.

    Returns:
        Batch job ID.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    batch_file_path = temp_dir / "batch_requests.jsonl"

    with open(batch_file_path, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    print(f"Created batch file: {batch_file_path}")

    with open(batch_file_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    print(f"Uploaded batch file: {batch_file.id}")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"Created batch job: {batch_job.id}")
    print(f"Status: {batch_job.status}\n")

    return batch_job.id


def save_batch_state(
    batch_job_id: str, metadata: Dict[str, Dict[str, Any]], state_file: Path
) -> None:
    """Save batch job state to file for later resumption.

    Args:
        batch_job_id: Batch job ID.
        metadata: Metadata dictionary with scene information.
        state_file: Path to save the state file.
    """
    state_data = {
        "batch_job_id": batch_job_id,
        "metadata": metadata,
    }

    state_file.parent.mkdir(parents=True, exist_ok=True)

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2, ensure_ascii=False)

    print(f"Saved batch job state to: {state_file}\n")


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
        Dictionary mapping custom_id to event description.
    """
    batch_job = client.batches.retrieve(batch_job_id)
    result_file_id = batch_job.output_file_id
    result_content = client.files.content(result_file_id).content

    results = {}
    for line in result_content.decode("utf-8").strip().split("\n"):
        result_obj = json.loads(line)
        custom_id = result_obj["custom_id"]
        event_description = result_obj["response"]["body"]["choices"][0]["message"][
            "content"
        ].strip()
        results[custom_id] = event_description

    print(f"Retrieved {len(results)} results\n")
    return results


def create_memory_tuples_from_results(
    metadata: Dict[str, Dict[str, Any]], results: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Create memory tuples from batch results.

    Args:
        metadata: Metadata dictionary with scene information.
        results: Dictionary mapping custom_id to event description.

    Returns:
        Dictionary mapping episode_id to list of memory tuples.
    """
    episodes_data: Dict[str, List[Dict[str, Any]]] = {}

    for custom_id, scene_meta in metadata.items():
        episode_id = scene_meta["episode_id"]

        if episode_id not in episodes_data:
            episodes_data[episode_id] = []

        event_description = results.get(
            custom_id,
            f"Scene at {scene_meta['location']} with {', '.join(scene_meta['subjects'][:3])}",
        )

        memory_tuple = {
            "label": custom_id,
            "time": {
                "start": scene_meta["start_time"],
                "end": scene_meta["end_time"],
                "duration_seconds": round(scene_meta["duration_seconds"], 2),
            },
            "location": scene_meta["location"],
            "subjects": scene_meta["subjects"],
            "event": event_description,
        }

        episodes_data[episode_id].append(memory_tuple)

    return episodes_data


def find_video_file(episode_id: str, videos_dir: Path) -> Optional[Path]:
    """Find video file for given episode.

    Args:
        episode_id: Episode identifier.
        videos_dir: Base videos directory.

    Returns:
        Path to video file, or None if not found.
    """
    # Format is "0101" season followed by episode
    season_num = episode_id[:2]
    episode_num = episode_id[2:]

    pattern = (
        videos_dir / f"season_{int(season_num)}" / f"episode_{int(episode_num)}.mp4"
    )

    return pattern if pattern.exists() else None


def extract_video_clip(
    video_path: Path, start_time: str, end_time: str, output_path: Path
) -> bool:
    """Extract video clip using ffmpeg.

    Args:
        video_path: Path to source video file.
        start_time: Start timestamp in SRT format.
        end_time: End timestamp in SRT format.
        output_path: Path for output clip.

    Returns:
        True if successful, False otherwise.
    """
    try:
        start_seconds = srt_time_to_seconds(start_time)
        end_seconds = srt_time_to_seconds(end_time)
        duration = end_seconds - start_seconds
        start_ffmpeg = seconds_to_ffmpeg_time(start_seconds)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            start_ffmpeg,
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            str(output_path),
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False


def extract_clips_and_save_tuples(
    episodes_data: Dict[str, List[Dict[str, Any]]],
    output_tuples_dir: Path,
    output_clips_dir: Path,
    videos_dir: Path,
) -> Tuple[int, int]:
    """Extract video clips and save memory tuples to files.

    Args:
        episodes_data: Dictionary mapping episode_id to list of memory tuples.
        output_tuples_dir: Directory for memory tuple JSON files.
        output_clips_dir: Directory for video clips.
        videos_dir: Base directory for videos.

    Returns:
        Tuple of (total_clips_extracted, total_clips_failed).
    """
    total_clips = 0
    total_failed = 0

    print(f"Extracting clips and saving tuples for {len(episodes_data)} episodes...")

    for episode_id, memory_tuples in sorted(episodes_data.items()):
        video_path = find_video_file(episode_id, videos_dir)
        if not video_path:
            print(
                f"Warning: Video file not found for {episode_id}, skipping clip extraction"
            )

        clips_extracted = 0
        clips_failed = 0

        for memory_tuple in memory_tuples:
            if video_path and memory_tuple["time"]["start"]:
                clip_filename = f"{memory_tuple['label']}.mp4"
                clip_path = output_clips_dir / episode_id / clip_filename

                if extract_video_clip(
                    video_path,
                    memory_tuple["time"]["start"],
                    memory_tuple["time"]["end"],
                    clip_path,
                ):
                    memory_tuple["video_clip_path"] = str(
                        clip_path.relative_to(output_clips_dir.parent)
                    )
                    clips_extracted += 1
                else:
                    clips_failed += 1

        output_data = {
            "episode": episode_id,
            "total_scenes": len(memory_tuples),
            "memory_tuples": memory_tuples,
        }

        output_file = output_tuples_dir / f"{episode_id}_memory_tuples.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        total_clips += clips_extracted
        total_failed += clips_failed

        print(
            f"  {episode_id}: {len(memory_tuples)} scenes, {clips_extracted} clips extracted"
        )

    print()
    return total_clips, total_failed


def process_all_episodes(
    synced_dir: Path,
    output_tuples_dir: Path,
    output_clips_dir: Path,
    videos_dir: Path,
    client: OpenAI,
) -> str:
    """Process all episodes - Stage 1: Submit batch job and save state.

    Args:
        synced_dir: Directory containing synced JSON files.
        output_tuples_dir: Directory for memory tuple JSON files.
        output_clips_dir: Directory for video clips.
        videos_dir: Base directory for videos.
        client: OpenAI client instance.

    Returns:
        Batch job ID.
    """
    print(f"\n{'='*80}")
    print("Episodic Memory Processing using OpenAI Batch API - Stage 1")
    print(f"{'='*80}\n")

    unique_locations = extract_unique_locations(synced_dir)

    location_mappings = generate_location_mappings_with_llm(unique_locations)

    batch_requests, metadata = prepare_batch_requests(
        synced_dir,
        location_mappings,
    )

    temp_dir = Path("./data/temp")
    batch_job_id = submit_batch_job(client, batch_requests, temp_dir)

    # Save state for resumption
    state_file = temp_dir / "batch_state.json"
    save_batch_state(batch_job_id, metadata, state_file)

    print("STAGE 1 COMPLETE:")
    print(f"Batch job ID: {batch_job_id}")
    print(f"Total scenes to process: {len(metadata)}")
    print(f"State saved to: {state_file}")

    return batch_job_id


def resume_process_all_episodes(
    output_tuples_dir: Path,
    output_clips_dir: Path,
    videos_dir: Path,
    client: OpenAI,
    state_file: Path,
) -> None:
    """Process all episodes - Stage 2: Resume from batch job completion.

    Args:
        output_tuples_dir: Directory for memory tuple JSON files.
        output_clips_dir: Directory for video clips.
        videos_dir: Base directory for videos.
        client: OpenAI client instance.
        state_file: Path to the saved batch state file.
    """
    print(f"\n{'='*80}")
    print("Episodic Memory Processing using OpenAI Batch API - Stage 2")
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

    episodes_data = create_memory_tuples_from_results(metadata, results)

    total_clips, total_failed = extract_clips_and_save_tuples(
        episodes_data, output_tuples_dir, output_clips_dir, videos_dir
    )

    print("PROCESSING SUMMARY:")
    print(f"Episodes processed: {len(episodes_data)}")
    print(f"Total scenes: {len(metadata)}")
    print(f"Total clips extracted: {total_clips}")
    print(f"Total clips failed: {total_failed}")
    print(f"Output directory: {output_tuples_dir}")


def process_data_pipeline() -> None:
    """Process pipeline for preprocessed data - Stage 1: Submit batch job.

    Pipeline stages:
    - Stage 1: Prepare and submit batch job for event descriptions.
    - Use resume_processing_pipeline() for Stage 2 after batch job completes.
    """
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data_dir = Path("./data")
    synced_dir = data_dir / "subtitles" / "synced"
    videos_dir = data_dir / "videos"
    output_tuples_dir = data_dir / "memory_tuples"
    output_clips_dir = data_dir / "clips"

    if not synced_dir.exists():
        raise FileNotFoundError(f"Synced subtitles directory not found: {synced_dir}")

    output_tuples_dir.mkdir(parents=True, exist_ok=True)
    output_clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_all_episodes(
            synced_dir, output_tuples_dir, output_clips_dir, videos_dir, client
        )
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


def resume_process_data_pipeline() -> None:
    """Resume processing pipeline - Stage 2: Process batch results and extract clips.

    This function loads the saved batch job state and continues processing
    once the batch job has completed.
    """
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data_dir = Path("./data")
    videos_dir = data_dir / "videos"
    output_tuples_dir = data_dir / "memory_tuples"
    output_clips_dir = data_dir / "clips"
    state_file = data_dir / "temp" / "batch_state.json"

    output_tuples_dir.mkdir(parents=True, exist_ok=True)
    output_clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        resume_process_all_episodes(
            output_tuples_dir, output_clips_dir, videos_dir, client, state_file
        )
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
