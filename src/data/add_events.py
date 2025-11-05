"""
Event Extraction and Data Enhancement Script

This script adds atomic event annotations to existing episode JSON files.
It extracts events from scenes using an LLM (OpenAI API via litellm) and saves
enhanced JSON files with the original structure preserved plus an 'events' array per scene.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import litellm
from textwrap import dedent
from tqdm import tqdm


@dataclass
class Event:
    """Represents an atomic event in the narrative."""

    event_id: str
    event_description: str
    involved_subjects: List[str]
    location: str
    timestamp: str
    episode_clip: str = ""


class LLMEventExtractor:
    """Extracts atomic events from scenes using OpenAI API via litellm."""

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Initialize the LLM event extractor.

        Args:
            model: Model name to use (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
        """
        self.model = model
        print(f"Initialized LLM Event Extractor with model: {model}")

    def extract_events(
        self,
        scene: Dict[str, Any],
        episode_id: str,
        scene_idx: int,
        max_events: int = 10,
    ) -> List[Event]:
        """
        Extract atomic events from a scene using LLM.

        Args:
            scene: Scene dictionary with dialogue, location, subjects, etc.
            episode_id: Episode identifier
            scene_idx: Scene index
            max_events: Maximum number of events to extract (default: 10)

        Returns:
            List of Event objects extracted from the scene
        """
        prompt = self._create_extraction_prompt(scene, max_events)

        response = self._generate_response(prompt)

        events = self._parse_events_from_response(
            response, scene, episode_id, scene_idx
        )

        return events[:max_events]

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

            Format your response as a JSON array (output ONLY the JSON, nothing else):
            [
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

            Output only the JSON array, no additional text."""
        )

        return prompt

    def _generate_response(self, prompt: str) -> str:
        """
        Generate response from LLM using litellm.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The generated text response from the LLM
        """
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                # max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Warning: Error generating response from LLM: {e}")
            return "[]"

    def _parse_events_from_response(
        self, response: str, scene: Dict[str, Any], episode_id: str, scene_idx: int
    ) -> List[Event]:
        """
        Parse events from LLM response.

        Args:
            response: The raw text response from the LLM
            scene: Scene dictionary for context
            episode_id: Episode identifier for event ID generation
            scene_idx: Scene index for event ID generation

        Returns:
            List of parsed Event objects
        """
        events: List[Event] = []

        try:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                events_data = json.loads(json_match.group())
            else:
                print(
                    f"Warning: Could not find JSON array in response for scene {scene_idx}"
                )
                return []

            if not isinstance(events_data, list):
                print(f"Warning: Response is not a JSON array for scene {scene_idx}")
                return []

            scene_subjects = set(scene.get("subjects", []))

            for idx, event_data in enumerate(events_data):
                if not isinstance(event_data, dict):
                    continue

                event_desc = event_data.get("event_description", "").strip()
                if not event_desc:
                    continue

                event_id = f"{episode_id}_scene_{scene_idx:03d}_event_{idx+1:03d}"

                raw_subjects = event_data.get("involved_subjects", [])
                if not isinstance(raw_subjects, list):
                    raw_subjects = []

                involved_subjects = [
                    s.strip() for s in raw_subjects if s.strip() in scene_subjects
                ]

                if not involved_subjects:
                    for subject in scene_subjects:
                        if subject.lower() in event_desc.lower():
                            involved_subjects.append(subject)

                start = event_data.get(
                    "start_offset", scene.get("clip_start", "00:00:00")
                )
                end = event_data.get("end_offset", scene.get("clip_end", "00:00:00"))
                timestamp = f"{start}-{end}"

                context_label = scene.get("context_label", "")
                if context_label:
                    parts = context_label.split("_")
                    if len(parts) >= 3:
                        episode_clip = f"{parts[0]}_scene_{parts[2]}"
                    else:
                        episode_clip = context_label
                else:
                    episode_clip = scene.get("episode_clip", "")

                event = Event(
                    event_id=event_id,
                    event_description=event_desc,
                    involved_subjects=involved_subjects,
                    location=scene.get("location", ""),
                    timestamp=timestamp,
                    episode_clip=episode_clip,
                )
                events.append(event)

        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error for scene {scene_idx}: {e}")
            return []
        except Exception as e:
            print(
                f"Warning: Unexpected error parsing events for scene {scene_idx}: {e}"
            )
            return []

        return events


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


def add_events_to_episode(
    episode_data: Dict[str, Any],
    llm_extractor: LLMEventExtractor,
    max_events_per_scene: int = 10,
) -> Dict[str, Any]:
    """
    Add events to an episode's scenes.

    Args:
        episode_data: Episode data dictionary containing scenes
        llm_extractor: LLM event extractor instance
        max_events_per_scene: Maximum events to extract per scene (default: 10)

    Returns:
        Enhanced episode data with events added to each scene
    """
    episode_id = episode_data.get("episode", "unknown")
    scenes = episode_data.get("scenes", [])

    total_events = 0

    scene_iterator = tqdm(scenes, desc="Extracting events")

    for scene_idx, scene in enumerate(scene_iterator):
        events = llm_extractor.extract_events(
            scene, episode_id, scene_idx, max_events=max_events_per_scene
        )
        scene["events"] = [asdict(e) for e in events]
        total_events += len(events)

    return episode_data


def process_single_file(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-4o",
    max_events_per_scene: int = 10,
    overwrite: bool = False,
) -> bool:
    """
    Process a single JSON file to extract and add events.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        model: Model name to use (e.g., "gpt-4o", "gpt-4-turbo")
        max_events_per_scene: Maximum events to extract per scene (default: 10)
        overwrite: Whether to overwrite existing output file (default: False)

    Returns:
        True if processing was successful, False otherwise
    """
    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        print("Set overwrite=True to replace it")
        return False

    try:
        llm_extractor = LLMEventExtractor(model=model)

        episode_data = load_episode_data(input_path)

        enhanced_data = add_events_to_episode(
            episode_data,
            llm_extractor,
            max_events_per_scene=max_events_per_scene,
        )

        save_json(enhanced_data, output_path)
        print(f"Saved: {output_path}\n")

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}\n")
        return False


def generate_events_from_subtitles(
    input_dir: Path,
    output_dir: Path,
    model: str = "gpt-4o",
    max_events_per_scene: int = 10,
    max_files: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = True,
) -> Dict[str, int]:
    """
    Process all JSON files in a directory to extract and add events.

    Args:
        input_dir: Input directory path containing JSON files
        output_dir: Output directory path for enhanced JSON files
        model: Model name to use (e.g., "gpt-4o", "gpt-4-turbo")
        max_events_per_scene: Maximum events to extract per scene (default: 10)
        max_files: Maximum number of files to process, None for all (default: None)
        overwrite: Whether to overwrite existing output files (default: False)
        resume: Whether to skip already processed files (default: True)

    Returns:
        Dictionary containing processing statistics:
            - total_files: Total number of files found
            - processed: Number of successfully processed files
            - skipped: Number of skipped files
            - failed: Number of failed files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if max_files:
        json_files = json_files[:max_files]

    print(f"\n{'='*70}")
    print(f"Processing {len(json_files)} files from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    llm_extractor = LLMEventExtractor(model=model)

    stats = {"total_files": len(json_files), "processed": 0, "skipped": 0, "failed": 0}

    if resume:
        processed_files = set()
        for f in output_dir.glob("*_with_events.json"):
            episode_id = f.stem.replace("_with_events", "")
            processed_files.add(episode_id)

        if processed_files:
            print(f"Resume mode: Found {len(processed_files)} already processed files")
            json_files = [f for f in json_files if f.stem not in processed_files]
            stats["skipped"] = len(processed_files)
            print(f"Will process {len(json_files)} remaining files\n")

    for idx, json_file in enumerate(json_files, 1):
        print(f"[{idx}/{len(json_files)}] Processing: {json_file.name}")

        output_filename = f"{json_file.stem}_with_events.json"
        output_path = output_dir / output_filename

        try:
            episode_data = load_episode_data(json_file)

            enhanced_data = add_events_to_episode(
                episode_data,
                llm_extractor,
                max_events_per_scene=max_events_per_scene,
            )

            save_json(enhanced_data, output_path)
            print(f"Saved: {output_path}\n")

            stats["processed"] += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}\n")
            stats["failed"] += 1

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total files:     {stats['total_files']}")
    print(f"Processed:       {stats['processed']}")
    print(f"Skipped:         {stats['skipped']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Output dir:      {output_dir}")
    print(f"{'='*70}\n")

    return stats


if __name__ == "__main__":
    print("This script is meant to be imported and used as a library.")
    print("\nExample usage:")
    print(
        """
    from pathlib import Path
    from add_events import process_single_file, generate_events_from_subtitles

    # Process single file
    process_single_file(
        input_path=Path("data/0102.json"),
        output_path=Path("output/0102_with_events.json"),
        model="gpt-4o"
    )

    # Process directory
    generate_events_from_subtitles(
        input_dir=Path("data/annotated_tuples"),
        output_dir=Path("output_with_events"),
        model="gpt-4o",
        max_events_per_scene=10
    )
    """
    )
