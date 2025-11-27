"""
Pipeline orchestration for CS707 G5 Project.
"""

from data.add_events import generate_events_from_subtitles  # type: ignore[import-not-found]
from data.generate_qa import generate_qa_from_events  # type: ignore[import-not-found]
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import sys
import json

load_dotenv()


def _find_metadata_file_for_batch(batch_id: str, search_dir: Path) -> Optional[Path]:
    """
    Find the metadata file path for a given batch_id by searching batch_info files.

    Args:
        batch_id: OpenAI batch job ID
        search_dir: Directory to search for batch_info files

    Returns:
        Path to metadata file if found, None otherwise
    """
    if not search_dir.exists():
        return None

    for batch_info_file in search_dir.glob("batch_info_*.json"):
        try:
            with open(batch_info_file, "r", encoding="utf-8") as f:
                batch_info = json.load(f)
                if batch_info.get("batch_id") == batch_id:
                    metadata_file = batch_info.get("metadata_file")
                    if metadata_file:
                        return Path(metadata_file)
        except (json.JSONDecodeError, IOError):
            continue

    return None


def events_generation() -> Dict[str, Any]:
    """
    Submit batch jobs for event extraction from subtitle files.

    Returns:
        Dictionary containing batch information including batch_id and metadata_file
    """
    result = generate_events_from_subtitles(
        input_dir=Path("main/subtitles/annotated_tuples/"),
        output_dir=Path("data/subtitles_with_events/"),
        model="gpt-4.1-nano",
        max_events_per_scene=10,
    )

    return result


def qa_generation() -> None:
    """
    Download batch results (if batch_id provided), process events, and generate QA samples.

    Command-line arguments (via sys.argv):
        batch_id: OpenAI batch job ID to download and process results first
        metadata_file: (Optional) Path to metadata JSON file from batch submission

    Usage:
        poe run-qa-generation                          # QA generation only (no batch download)
        poe run-qa-generation <batch_id>               # Download batch results + QA generation (auto-find metadata)
        poe run-qa-generation <batch_id> <metadata>    # Download batch results + QA generation (explicit metadata)
    """
    batch_id: Optional[str] = None
    metadata_file: Optional[Path] = None

    args = sys.argv[1:]
    if len(args) >= 1:
        batch_id = args[0]

        if len(args) >= 2:
            metadata_file = Path(args[1])
        else:
            batch_info_dir = Path("data/subtitles_with_events/")
            metadata_file = _find_metadata_file_for_batch(batch_id, batch_info_dir)

            if metadata_file is None:
                print(f"Error: Could not find metadata file for batch_id: {batch_id}")
                print(f"Searched in: {batch_info_dir}")
                print("\nMake sure you run this from the project root directory.")
                print("Alternatively, provide the metadata file path explicitly:")
                print("  poe run-qa-generation <batch_id> <metadata_file>")
                sys.exit(1)

    print("\n" + "=" * 70)
    print("QA GENERATION CONFIGURATION")
    print("=" * 70)
    if batch_id:
        print(f"Batch ID: {batch_id}")
        print(f"Metadata File: {metadata_file}")
        print("Mode: Download batch results + Generate QA datasets")
    else:
        print("Mode: Generate QA datasets only (no batch download)")
    print("=" * 70 + "\n")

    generate_qa_from_events(
        input_dir=Path("data/subtitles_with_events/"),
        output_dir=Path("data/qa_output/"),
        max_episode_span=3,
        temporal_questions_per_span=2,
        max_concurrent_files=5,
        batch_id=batch_id,
        metadata_file=metadata_file,
    )
