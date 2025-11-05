"""
Pipeline orchestration for CS707 G5 Project.
"""

from pathlib import Path
from data.add_events import generate_events_from_subtitles  # type: ignore[import-not-found]
from data.generate_qa import generate_qa_from_events  # type: ignore[import-not-found]


def events_generation() -> None:
    generate_events_from_subtitles(
        input_dir=Path("data/annotated_tuples"),
        output_dir=Path("output_with_events"),
        model="gpt-4o",
        max_events_per_scene=10,
    )


def qa_generation() -> None:
    generate_qa_from_events()
