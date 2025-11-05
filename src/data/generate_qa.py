"""
Question Generation from Enhanced Episode Data

This script generates QA datasets from episode JSON files that already have events.
Use this after running add_events.py to separate event extraction from question generation.

This script also handles downloading and processing batch results from OpenAI's Batch API
when events are extracted via batch processing.
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from openai import OpenAI


@dataclass
class Event:
    """Represents an atomic event in the narrative."""

    event_id: str
    event_description: str
    involved_subjects: List[str]
    location: str
    timestamp: str
    episode_clip: str = ""


@dataclass
class Question:
    """
    Represents a generated question with all associated metadata.

    Attributes:
        question_id: Unique identifier for the question
        question_type: Type of question (e.g., 'single target recall', 'temporal', 'boolean')
        subcategory: More specific categorization within the question type
        cues: Input cues provided in the question (e.g., 'E+S', 'L+E')
        target: What the question is asking for (e.g., 'L' for location, 'S' for subject)
        question: The actual question text
        options: List of multiple choice options
        answer: The correct answer letter (A, B, C, or D)
        answer_index: Zero-based index of correct answer
        explanation: Optional explanation of the answer
        related_events: List of event IDs related to this question
        episode_span: List of episode IDs covered by this question
        clip_span: List of video clips related to this question
    """

    question_id: str
    question_type: str
    subcategory: str
    cues: str = ""
    target: str = ""
    question: str = ""
    options: Optional[List[str]] = None
    answer: str = ""
    answer_index: int = -1
    explanation: str = ""
    related_events: Optional[List[str]] = None
    episode_span: Optional[List[str]] = None
    clip_span: Optional[List[str]] = None

    def __post_init__(self):
        if self.options is None:
            self.options = []
        if self.related_events is None:
            self.related_events = []
        if self.episode_span is None:
            self.episode_span = []
        if self.clip_span is None:
            self.clip_span = []


def load_episode_data(json_path: str) -> Dict[str, Any]:
    """Load episode data from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, output_path: str):
    """Save data to JSON file with proper formatting."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_statistics(questions: List[Question]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for generated questions.

    Analyzes question type distribution, subcategory distribution, cue-target mappings,
    episode/clip spans, and answer distribution to ensure balanced QA generation.

    Args:
        questions: List of Question objects to analyze

    Returns:
        Dictionary with statistics including total_questions, question_type_distribution,
        subcategory_distribution, cue_distribution, target_distribution,
        episode_span_distribution, clip_span_distribution, and answer_distribution
    """
    stats: Dict[str, Any] = {
        "total_questions": len(questions),
        "question_type_distribution": defaultdict(int),
        "subcategory_distribution": defaultdict(int),
        "cue_distribution": defaultdict(int),
        "target_distribution": defaultdict(int),
        "episode_span_distribution": defaultdict(int),
        "clip_span_distribution": defaultdict(int),
        "answer_distribution": defaultdict(int),
    }

    for q in questions:
        stats["question_type_distribution"][q.question_type] += 1
        stats["subcategory_distribution"][q.subcategory] += 1

        if q.cues:
            stats["cue_distribution"][f"{q.cues} → {q.target}"] += 1
        if q.target:
            stats["target_distribution"][q.target] += 1
        elif "temporal" in q.question_type:
            stats["target_distribution"]["Temporal"] += 1
        elif "boolean" in q.question_type:
            stats["target_distribution"]["Boolean"] += 1

        if q.episode_span:
            span_count = len(q.episode_span)
            stats["episode_span_distribution"][f"{span_count} episode(s)"] += 1

        if q.clip_span:
            clip_count = len(q.clip_span)
            stats["clip_span_distribution"][f"{clip_count} clip(s)"] += 1

        if isinstance(q.answer, str) and len(q.answer) == 1 and q.answer in "ABCD":
            stats["answer_distribution"][q.answer] += 1
        elif q.answer_index >= 0 and q.answer_index < 4:
            answer_letter = chr(65 + q.answer_index)
            stats["answer_distribution"][answer_letter] += 1

    return {
        "total_questions": stats["total_questions"],
        "question_type_distribution": {
            k: v for k, v in stats["question_type_distribution"].items()
        },
        "subcategory_distribution": {
            k: v for k, v in stats["subcategory_distribution"].items()
        },
        "cue_distribution": {k: v for k, v in stats["cue_distribution"].items()},
        "target_distribution": {k: v for k, v in stats["target_distribution"].items()},
        "episode_span_distribution": {
            k: v for k, v in stats["episode_span_distribution"].items()
        },
        "clip_span_distribution": {
            k: v for k, v in stats["clip_span_distribution"].items()
        },
        "answer_distribution": {k: v for k, v in stats["answer_distribution"].items()},
    }


def print_statistics_report(stats: Dict[str, Any], episode_id: str = ""):
    """
    Print a formatted statistics report to console.

    Args:
        stats: Statistics dictionary from calculate_statistics
        episode_id: Optional episode ID for the report header
    """
    print("\n" + "=" * 70)
    if episode_id:
        print(f"QUESTION GENERATION STATISTICS - Episode {episode_id}")
    else:
        print("QUESTION GENERATION STATISTICS")
    print("=" * 70)

    print(f"\nTotal Questions: {stats['total_questions']}")

    print("\n--- Question Type Distribution ---")
    for q_type, count in sorted(stats["question_type_distribution"].items()):
        percentage = (count / stats["total_questions"]) * 100
        print(f"  {q_type:<45} {count:>4} ({percentage:>5.1f}%)")

    print("\n--- Subcategory Distribution ---")
    for subcat, count in sorted(stats["subcategory_distribution"].items()):
        percentage = (count / stats["total_questions"]) * 100
        print(f"  {subcat:<45} {count:>4} ({percentage:>5.1f}%)")

    if stats.get("cue_distribution"):
        print("\n--- Cue-Target Distribution ---")
        for cue_target, count in sorted(stats["cue_distribution"].items()):
            percentage = (count / stats["total_questions"]) * 100
            print(f"  {cue_target:<45} {count:>4} ({percentage:>5.1f}%)")

    if stats.get("target_distribution"):
        print("\n--- Target Distribution ---")
        for target, count in sorted(stats["target_distribution"].items()):
            percentage = (count / stats["total_questions"]) * 100
            print(f"  {target:<45} {count:>4} ({percentage:>5.1f}%)")

    if stats.get("answer_distribution"):
        print("\n--- Answer Distribution (A/B/C/D) ---")
        for answer, count in sorted(stats["answer_distribution"].items()):
            percentage = (count / stats["total_questions"]) * 100
            print(f"  {answer:<45} {count:>4} ({percentage:>5.1f}%)")

    if stats.get("episode_span_distribution"):
        print("\n--- Episode Span Distribution ---")
        for span, count in sorted(stats["episode_span_distribution"].items()):
            percentage = (count / stats["total_questions"]) * 100
            print(f"  {span:<45} {count:>4} ({percentage:>5.1f}%)")

    if stats.get("clip_span_distribution"):
        print("\n--- Clip Span Distribution ---")
        for span, count in sorted(stats["clip_span_distribution"].items()):
            percentage = (count / stats["total_questions"]) * 100
            print(f"  {span:<45} {count:>4} ({percentage:>5.1f}%)")

    print("=" * 70 + "\n")


def download_and_process_batch_results(
    batch_id: str,
    metadata_file: Path,
    output_dir: Path,
) -> Dict[str, int]:
    """
    Download batch results from OpenAI and process them into final JSON files.

    Retrieves completed batch results from OpenAI, parses the event extraction outputs,
    and merges them back into the original episode JSON files. Creates enhanced episode
    files with events added to each scene. Only processes batches with completed status.

    Args:
        batch_id: OpenAI batch job ID
        metadata_file: Path to the metadata JSON file created during batch submission
        output_dir: Output directory for processed episode JSON files

    Returns:
        Dictionary containing processing statistics:
            - total_scenes: Total number of scenes processed
            - episodes_created: Number of episode files created
            - failed: Number of failed requests
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"\n{'='*70}")
    print("DOWNLOADING BATCH RESULTS")
    print(f"{'='*70}")
    print(f"Batch ID: {batch_id}")

    batch = client.batches.retrieve(batch_id)
    print(f"Status: {batch.status}")

    if batch.status != "completed":
        print(f"\nBatch is not completed yet. Current status: {batch.status}")
        print("Please wait and try again later.")
        return {"total_scenes": 0, "episodes_created": 0, "failed": 0}

    if not batch.output_file_id:
        print("Error: No output file available")
        return {"total_scenes": 0, "episodes_created": 0, "failed": 0}

    print(f"Downloading results from file: {batch.output_file_id}")
    results_content = client.files.content(batch.output_file_id)
    results_text = results_content.read().decode("utf-8")

    results = []
    for line in results_text.strip().split("\n"):
        if line:
            results.append(json.loads(line))

    print(f"Downloaded {len(results)} results")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_map = {req["custom_id"]: req for req in metadata["requests"]}

    episodes_data = {}
    stats = {"total_scenes": 0, "episodes_created": 0, "failed": 0}

    for result in tqdm(results, desc="Processing results"):
        custom_id = result["custom_id"]

        if result["response"]["status_code"] != 200:
            print(f"Warning: Request {custom_id} failed")
            stats["failed"] += 1
            continue

        req_metadata = metadata_map.get(custom_id)
        if not req_metadata:
            print(f"Warning: No metadata found for {custom_id}")
            continue

        json_file = Path(req_metadata["json_file"])
        scene_idx = req_metadata["scene_idx"]

        if json_file not in episodes_data:
            episodes_data[json_file] = load_episode_data(str(json_file))

        response_body = result["response"]["body"]
        events_json = response_body["choices"][0]["message"]["content"]
        events_data = json.loads(events_json)

        episodes_data[json_file]["scenes"][scene_idx]["events"] = events_data["events"]
        stats["total_scenes"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file, episode_data in tqdm(
        episodes_data.items(), desc="Saving episode files"
    ):
        output_filename = f"{json_file.stem}_with_events.json"
        output_path = output_dir / output_filename
        save_json(episode_data, str(output_path))
        stats["episodes_created"] += 1

    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total scenes processed: {stats['total_scenes']}")
    print(f"Episodes created:       {stats['episodes_created']}")
    print(f"Failed requests:        {stats['failed']}")
    print(f"Output directory:       {output_dir}")
    print(f"{'='*70}\n")

    return stats


class QuestionGenerator:
    """
    Generates various types of questions from narrative events.

    This class handles generation of multiple question types:
    - Single target recall (E+S→L, L+E→S, L+S→E): ~60% of questions
    - Boolean verification: ~20% of questions
    - Temporal ordering: ~15% of questions
    - Latest event retrieval: ~5% of questions
    - Temporal single cue retrieval (location-based ordering): 2 per episode
    """

    def __init__(self):
        self.question_counter = 0

    def generate_all_questions(
        self,
        episode_data: Dict[str, Any],
        events_by_episode: Dict[str, List[Event]],
        current_episode_id: str,
        max_episode_span: int = 3,
        temporal_questions_per_span: int = 2,
    ) -> List[Question]:
        """
        Generate all question types for an episode.

        Args:
            episode_data: Episode data with scenes and events
            events_by_episode: Dictionary mapping episode_id to events for temporal questions
            current_episode_id: Current episode being processed
            max_episode_span: Maximum number of episodes to span for temporal questions
            temporal_questions_per_span: Number of temporal questions per span range

        Returns:
            List of Question objects covering all question types
        """
        questions = []
        episode_id = episode_data["episode"]

        events = []
        for scene in episode_data["scenes"]:
            scene_events = scene.get("events", [])
            for e in scene_events:
                if isinstance(e, dict):
                    events.append(Event(**e))
                else:
                    events.append(e)

        if not events:
            print(f"Warning: No events found in episode {episode_id}")
            return questions

        str_questions = self._generate_single_target_recall(events, episode_id)
        questions.extend(str_questions)

        bool_questions = self._generate_boolean_questions(events, episode_id)
        questions.extend(bool_questions)

        temporal_general_questions = self._generate_temporal_ordering(
            events_by_episode,
            current_episode_id,
            max_episode_span,
            temporal_questions_per_span,
        )
        questions.extend(temporal_general_questions)

        temporal_latest_questions = self._generate_latest_event_retrieval(
            events, episode_id
        )
        questions.extend(temporal_latest_questions)

        temporal_location_questions = self._generate_temporal_single_cue_retrieval(
            events, current_episode_id, num_questions=2, num_events=4
        )
        questions.extend(temporal_location_questions)

        return questions

    def _generate_single_target_recall(
        self, events: List[Event], episode_id: str
    ) -> List[Question]:
        """
        Generate single target recall questions (E+S→L, L+E→S, L+S→E).

        Creates three types of recall questions for each event:
        1. Given event and subject, recall location
        2. Given location and event, recall subject
        3. Given location and subject, recall event
        """
        questions = []

        all_locations = list(set(e.location for e in events if e.location))

        for event in events:
            if not event.location:
                continue

            q = self._create_location_recall_question(event, episode_id, all_locations)
            if q:
                questions.append(q)

            q = self._create_subject_recall_question(event, episode_id, events)
            if q:
                questions.append(q)

            q = self._create_event_recall_question(event, episode_id, events)
            if q:
                questions.append(q)

        return questions

    def _create_location_recall_question(
        self, event: Event, episode_id: str, all_locations: List[str]
    ) -> Optional[Question]:
        """
        Create E+S→L (Event + Subject → Location) question.

        Generates distractors from other locations in the episode and randomizes
        answer position to avoid pattern bias.
        """
        if len(event.involved_subjects) == 0:
            return None

        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"

        subjects_str = " and ".join(event.involved_subjects)
        question_text = f"Where did {event.event_description.lower()} when {subjects_str} were present?"

        distractors = [loc for loc in all_locations if loc != event.location]
        options_raw = [event.location] + random.sample(
            distractors, min(3, len(distractors))
        )

        while len(options_raw) < 4:
            options_raw.append(f"Unknown location {len(options_raw)}")
        options_raw = options_raw[:4]

        answer_index = random.randint(0, 3)
        options: List[str] = [""] * 4
        options[answer_index] = event.location

        distractor_idx = 0
        for i in range(4):
            if not options[i]:
                options[i] = options_raw[1 + distractor_idx]
                distractor_idx += 1

        answer_letter = chr(65 + answer_index)

        return Question(
            question_id=question_id,
            question_type="single target recall",
            subcategory="location recall (event + subject)",
            cues="E+S",
            target="L",
            question=question_text,
            options=options,
            answer=answer_letter,
            answer_index=answer_index,
            related_events=[event.event_id],
            episode_span=[episode_id],
            clip_span=[event.episode_clip] if event.episode_clip else [],
        )

    def _create_subject_recall_question(
        self, event: Event, episode_id: str, all_events: List[Event]
    ) -> Optional[Question]:
        """
        Create L+E→S (Location + Event → Subject) question.

        Uses other subject combinations from the same location as distractors,
        along with variations of the correct answer.
        """
        if len(event.involved_subjects) == 0:
            return None

        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"

        question_text = f"Who was present at {event.location} when {event.event_description.lower()}?"

        same_location_events = [
            e
            for e in all_events
            if e.location == event.location and e.event_id != event.event_id
        ]

        correct_answer = ", ".join(sorted(event.involved_subjects))
        distractors = []

        for e in same_location_events[:3]:
            if e.involved_subjects:
                distractor = ", ".join(sorted(e.involved_subjects))
                if distractor != correct_answer:
                    distractors.append(distractor)

        if len(distractors) < 3 and len(event.involved_subjects) > 1:
            distractors.append(", ".join(event.involved_subjects[:-1]))

        options_raw = [correct_answer] + distractors[:3]

        while len(options_raw) < 4:
            options_raw.append(f"Unknown subjects {len(options_raw)}")
        options_raw = options_raw[:4]

        answer_index = random.randint(0, 3)
        options: List[str] = [""] * 4
        options[answer_index] = correct_answer

        distractor_idx = 0
        for i in range(4):
            if not options[i]:
                options[i] = (
                    options_raw[1 + distractor_idx]
                    if 1 + distractor_idx < len(options_raw)
                    else f"Unknown subjects {i}"
                )
                distractor_idx += 1

        answer_letter = chr(65 + answer_index)

        return Question(
            question_id=question_id,
            question_type="single target recall",
            subcategory="subject recall (location + event)",
            cues="L+E",
            target="S",
            question=question_text,
            options=options,
            answer=answer_letter,
            answer_index=answer_index,
            related_events=[event.event_id],
            episode_span=[episode_id],
            clip_span=[event.episode_clip] if event.episode_clip else [],
        )

    def _create_event_recall_question(
        self, event: Event, episode_id: str, all_events: List[Event]
    ) -> Optional[Question]:
        """
        Create L+S→E (Location + Subject → Event) question.

        Uses other events at the same location as distractors.
        """
        if len(event.involved_subjects) == 0:
            return None

        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"

        subjects_str = " and ".join(event.involved_subjects)
        question_text = (
            f"What happened at {event.location} when {subjects_str} were together?"
        )

        same_location_events = [
            e
            for e in all_events
            if e.location == event.location and e.event_id != event.event_id
        ]

        distractors = [e.event_description for e in same_location_events[:3]]

        options_raw = [event.event_description] + distractors

        while len(options_raw) < 4:
            options_raw.append(f"Unknown event {len(options_raw)}")
        options_raw = options_raw[:4]

        answer_index = random.randint(0, 3)
        options: List[str] = [""] * 4
        options[answer_index] = event.event_description

        distractor_idx = 0
        for i in range(4):
            if not options[i]:
                options[i] = (
                    options_raw[1 + distractor_idx]
                    if 1 + distractor_idx < len(options_raw)
                    else f"Unknown event {i}"
                )
                distractor_idx += 1

        answer_letter = chr(65 + answer_index)

        return Question(
            question_id=question_id,
            question_type="single target recall",
            subcategory="event recall (location + subject)",
            cues="L+S",
            target="E",
            question=question_text,
            options=options,
            answer=answer_letter,
            answer_index=answer_index,
            related_events=[event.event_id],
            episode_span=[episode_id],
            clip_span=[event.episode_clip] if event.episode_clip else [],
        )

    def _generate_boolean_questions(
        self, events: List[Event], episode_id: str
    ) -> List[Question]:
        """
        Generate boolean verification questions with negative examples.

        Creates false verification questions by pairing events with wrong locations
        or wrong subjects. Samples approximately 20% of events for boolean questions.
        """
        questions = []

        all_locations = list(set(e.location for e in events if e.location))
        all_subjects = list(set(s for e in events for s in e.involved_subjects))

        sampled_events = random.sample(
            events, min(len(events), max(1, len(events) // 5))
        )

        for event in sampled_events:
            wrong_locations = [loc for loc in all_locations if loc != event.location]
            if wrong_locations:
                self.question_counter += 1
                question_id = f"Q_{episode_id}_{self.question_counter:03d}"

                wrong_loc = random.choice(wrong_locations)
                question_text = f"Did {event.event_description.lower()} at {wrong_loc}?"

                answer_index = random.randint(0, 1)
                options = ["Yes", "No"] if answer_index == 1 else ["No", "Yes"]
                answer_letter = chr(65 + options.index("No"))

                questions.append(
                    Question(
                        question_id=question_id,
                        question_type="boolean",
                        subcategory="location verification",
                        question=question_text,
                        options=options,
                        answer=answer_letter,
                        answer_index=options.index("No"),
                        explanation=f"{event.event_description} occurred at {event.location}, not {wrong_loc}.",
                        related_events=[event.event_id],
                        episode_span=[episode_id],
                        clip_span=[event.episode_clip] if event.episode_clip else [],
                    )
                )

            absent_subjects = [
                s for s in all_subjects if s not in event.involved_subjects
            ]
            if absent_subjects:
                self.question_counter += 1
                question_id = f"Q_{episode_id}_{self.question_counter:03d}"

                wrong_subject = random.choice(absent_subjects)
                question_text = f"Was {wrong_subject} present when {event.event_description.lower()}?"

                answer_index = random.randint(0, 1)
                options = ["Yes", "No"] if answer_index == 1 else ["No", "Yes"]
                answer_letter = chr(65 + options.index("No"))

                questions.append(
                    Question(
                        question_id=question_id,
                        question_type="boolean",
                        subcategory="subject verification",
                        question=question_text,
                        options=options,
                        answer=answer_letter,
                        answer_index=options.index("No"),
                        explanation=f"{wrong_subject} was not present during this event.",
                        related_events=[event.event_id],
                        episode_span=[episode_id],
                        clip_span=[event.episode_clip] if event.episode_clip else [],
                    )
                )

        return questions

    def _generate_temporal_ordering(
        self,
        events_by_episode: Dict[str, List[Event]],
        current_episode_id: str,
        max_episode_span: int = 3,
        temporal_questions_per_span: int = 2,
    ) -> List[Question]:
        """
        Generate temporal ordering questions spanning different numbers of episodes.

        Creates questions that test understanding of event chronology. Spans range from
        single episode to max_episode_span episodes. Distributes event pairs evenly
        across the span to ensure diverse temporal reasoning.

        Args:
            events_by_episode: Dictionary mapping episode_id to events
            current_episode_id: Current episode being processed
            max_episode_span: Maximum number of episodes to span (1 to N)
            temporal_questions_per_span: Number of questions per span configuration

        Returns:
            List of temporal ordering questions
        """
        questions = []

        current_events = events_by_episode.get(current_episode_id, [])
        if len(current_events) < 2:
            return questions

        episode_ids = sorted(events_by_episode.keys())
        current_idx = (
            episode_ids.index(current_episode_id)
            if current_episode_id in episode_ids
            else 0
        )

        for span in range(1, max_episode_span + 1):
            start_idx = max(0, current_idx - span + 1)
            end_idx = min(len(episode_ids), current_idx + 1)
            relevant_episode_ids = episode_ids[start_idx:end_idx]

            if len(relevant_episode_ids) < span:
                continue

            span_events = []
            for ep_id in relevant_episode_ids:
                span_events.extend(events_by_episode.get(ep_id, []))

            sorted_events = sorted(span_events, key=lambda e: e.event_id)

            if len(sorted_events) < 2:
                continue

            num_questions = min(temporal_questions_per_span, len(sorted_events) - 1)

            for q_idx in range(num_questions):
                if len(sorted_events) > 3:
                    step = max(1, len(sorted_events) // (num_questions + 1))
                    i = min(q_idx * step, len(sorted_events) // 2)
                    j = min(i + step + q_idx, len(sorted_events) - 1)
                else:
                    i = 0
                    j = len(sorted_events) - 1

                event_a = sorted_events[i]
                event_b = sorted_events[j]

                self.question_counter += 1
                question_id = f"Q_{current_episode_id}_{self.question_counter:03d}"

                question_text = (
                    f"What happened before {event_b.event_description.lower()}?"
                )

                distractors = []
                for k, e in enumerate(sorted_events):
                    if k != i and e.event_id != event_b.event_id:
                        distractors.append(e.event_description)
                    if len(distractors) >= 3:
                        break

                options_raw = [event_a.event_description] + distractors[:3]

                while len(options_raw) < 4:
                    options_raw.append(f"Unknown event {len(options_raw)}")
                options_raw = options_raw[:4]

                answer_index = random.randint(0, 3)
                options: List[str] = [""] * 4
                options[answer_index] = event_a.event_description

                distractor_idx = 0
                for k in range(4):
                    if not options[k]:
                        options[k] = (
                            options_raw[1 + distractor_idx]
                            if 1 + distractor_idx < len(options_raw)
                            else f"Unknown event {k}"
                        )
                        distractor_idx += 1

                answer_letter = chr(65 + answer_index)

                episode_span = list(
                    set(
                        [
                            event_a.event_id.split("_scene_")[0],
                            event_b.event_id.split("_scene_")[0],
                        ]
                    )
                )

                clip_span = []
                if event_a.episode_clip:
                    clip_span.append(event_a.episode_clip)
                if (
                    event_b.episode_clip
                    and event_b.episode_clip != event_a.episode_clip
                ):
                    clip_span.append(event_b.episode_clip)

                questions.append(
                    Question(
                        question_id=question_id,
                        question_type="temporal: chronological ordering",
                        subcategory=f"event sequence ({span}-episode span)",
                        question=question_text,
                        options=options,
                        answer=answer_letter,
                        answer_index=answer_index,
                        related_events=[event_a.event_id, event_b.event_id],
                        episode_span=episode_span,
                        clip_span=clip_span,
                    )
                )

        return questions

    def _generate_latest_event_retrieval(
        self, events: List[Event], episode_id: str
    ) -> List[Question]:
        """
        Generate latest event retrieval questions.

        Creates two types of questions:
        1. Latest event at a specific location within the episode
        2. Latest event in the entire episode

        Uses earlier events as distractors.
        """
        questions = []

        if not events:
            return questions

        events_by_location = defaultdict(list)
        for event in events:
            if event.location:
                events_by_location[event.location].append(event)

        for location, loc_events in events_by_location.items():
            if len(loc_events) < 2:
                continue

            latest_event = loc_events[-1]

            self.question_counter += 1
            question_id = f"Q_{episode_id}_{self.question_counter:03d}"

            question_text = (
                f"What was the last event that happened at {location} in this episode?"
            )

            distractors = [e.event_description for e in loc_events[:-1]][-3:]

            options_raw = [latest_event.event_description] + distractors

            while len(options_raw) < 4:
                options_raw.append(f"Unknown event {len(options_raw)}")
            options_raw = options_raw[:4]

            answer_index = random.randint(0, 3)
            options: List[str] = [""] * 4
            options[answer_index] = latest_event.event_description

            distractor_idx = 0
            for i in range(4):
                if not options[i]:
                    options[i] = (
                        options_raw[1 + distractor_idx]
                        if 1 + distractor_idx < len(options_raw)
                        else f"Unknown event {i}"
                    )
                    distractor_idx += 1

            answer_letter = chr(65 + answer_index)

            questions.append(
                Question(
                    question_id=question_id,
                    question_type="temporal: latest event retrieval",
                    subcategory="location-based latest event",
                    question=question_text,
                    options=options,
                    answer=answer_letter,
                    answer_index=answer_index,
                    related_events=[latest_event.event_id],
                    episode_span=[episode_id],
                    clip_span=(
                        [latest_event.episode_clip] if latest_event.episode_clip else []
                    ),
                )
            )

            break

        if len(events) >= 2:
            latest_event = events[-1]

            self.question_counter += 1
            question_id = f"Q_{episode_id}_{self.question_counter:03d}"

            question_text = (
                "What was the most recent event that occurred in the episode?"
            )

            distractors = [e.event_description for e in events[:-1]][-3:]

            options_raw = [latest_event.event_description] + distractors

            while len(options_raw) < 4:
                options_raw.append(f"Unknown event {len(options_raw)}")
            options_raw = options_raw[:4]

            answer_index = random.randint(0, 3)
            options: List[str] = [""] * 4
            options[answer_index] = latest_event.event_description

            distractor_idx = 0
            for i in range(4):
                if not options[i]:
                    options[i] = (
                        options_raw[1 + distractor_idx]
                        if 1 + distractor_idx < len(options_raw)
                        else f"Unknown event {i}"
                    )
                    distractor_idx += 1

            answer_letter = chr(65 + answer_index)

            questions.append(
                Question(
                    question_id=question_id,
                    question_type="temporal: latest event retrieval",
                    subcategory="episode-wide latest event",
                    question=question_text,
                    options=options,
                    answer=answer_letter,
                    answer_index=answer_index,
                    related_events=[latest_event.event_id],
                    episode_span=[episode_id],
                    clip_span=(
                        [latest_event.episode_clip] if latest_event.episode_clip else []
                    ),
                )
            )

        return questions

    def _generate_temporal_single_cue_retrieval(
        self,
        events: List[Event],
        episode_id: str,
        num_questions: int = 2,
        num_events: int = 4,
    ) -> List[Question]:
        """
        Generate temporal single cue retrieval questions for location-based ordering.

        Asks users to order events that occurred at the same location chronologically.
        Prefers events from multiple scenes for better temporal diversity. Uses
        timestamp parsing to determine correct chronological order.

        Args:
            events: List of events from the current episode
            episode_id: Current episode ID
            num_questions: Number of questions to generate per episode
            num_events: Number of events per question to order

        Returns:
            List of temporal single cue retrieval questions
        """
        questions = []

        if not events:
            return questions

        events_data = []
        for event in events:
            parts = event.event_id.split("_")
            scene_num = parts[2] if len(parts) > 2 else "000"

            try:
                start_time = event.timestamp.split("-")[0].strip()
                h, m, s = map(int, start_time.split(":"))
                start_seconds = h * 3600 + m * 60 + s
            except Exception:
                start_seconds = 0

            events_data.append(
                {
                    "event_id": event.event_id,
                    "event_description": event.event_description,
                    "location": event.location,
                    "scene_id": f"{episode_id}_scene_{scene_num}",
                    "start_seconds": start_seconds,
                }
            )

        locations = defaultdict(list)
        for ev in events_data:
            if ev["location"]:
                locations[ev["location"]].append(ev)

        valid_locations = [
            loc
            for loc, evs in locations.items()
            if len(set(e["scene_id"] for e in evs)) > 1 and len(evs) >= num_events
        ]

        if not valid_locations:
            valid_locations = [
                loc for loc, evs in locations.items() if len(evs) >= num_events
            ]

        if not valid_locations:
            return questions

        for q_idx in range(min(num_questions, len(valid_locations))):
            location = (
                valid_locations[q_idx]
                if q_idx < len(valid_locations)
                else random.choice(valid_locations)
            )
            loc_events = locations[location]

            scenes = list({e["scene_id"] for e in loc_events})
            random.shuffle(scenes)
            chosen_events = []
            used_scenes = set()

            for e in random.sample(loc_events, min(len(loc_events), num_events * 2)):
                if e["scene_id"] not in used_scenes:
                    chosen_events.append(e)
                    used_scenes.add(e["scene_id"])
                if len(chosen_events) == num_events:
                    break

            if len(chosen_events) < num_events:
                remaining = [e for e in loc_events if e not in chosen_events]
                chosen_events += random.sample(
                    remaining, min(len(remaining), num_events - len(chosen_events))
                )

            if len(chosen_events) < 2:
                continue

            chosen_events = chosen_events[:num_events]

            labeled = {
                f"event_{i+1}": chosen_events[i]["event_description"]
                for i in range(len(chosen_events))
            }
            labeled_ids = {
                f"event_{i+1}": chosen_events[i]["event_id"]
                for i in range(len(chosen_events))
            }
            labeled_start = {
                f"event_{i+1}": chosen_events[i]["start_seconds"]
                for i in range(len(chosen_events))
            }

            correct_sequence = [
                label
                for label, _ in sorted(labeled_start.items(), key=lambda kv: kv[1])
            ]
            all_labels = list(labeled.keys())

            options_sequences = [correct_sequence.copy()]
            seen = {tuple(correct_sequence)}

            max_attempts = 50
            attempts = 0
            while len(options_sequences) < 4 and attempts < max_attempts:
                perm = random.sample(all_labels, len(all_labels))
                if tuple(perm) not in seen:
                    options_sequences.append(perm)
                    seen.add(tuple(perm))
                attempts += 1

            while len(options_sequences) < 4:
                perm = correct_sequence.copy()
                if len(perm) >= 2:
                    idx = random.randint(0, len(perm) - 2)
                    perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
                    if tuple(perm) not in seen:
                        options_sequences.append(perm)
                        seen.add(tuple(perm))
                    else:
                        options_sequences.append(perm)
                        break

            answer_index = random.randint(0, 3)

            final_options: List[List[str]] = [[]] * 4
            final_options[answer_index] = correct_sequence

            opt_idx = 0
            for i in range(4):
                if not final_options[i]:
                    if opt_idx + 1 < len(options_sequences):
                        final_options[i] = options_sequences[opt_idx + 1]
                        opt_idx += 1
                    else:
                        perm = correct_sequence.copy()
                        random.shuffle(perm)
                        final_options[i] = perm

            letters = ["A", "B", "C", "D"]
            mc_options = [
                f"{letters[i]}. {', '.join(final_options[i])}" for i in range(4)
            ]
            answer_letter = letters[answer_index]

            question_text = (
                f"Here are a series of {len(chosen_events)} events. Arrange them in the correct temporal order "
                f"in which they occurred at **{location}**.\n\n"
                + "\n".join([f'"{label}": {desc}' for label, desc in labeled.items()])
            )

            self.question_counter += 1
            question_id = f"Q_{episode_id}_{self.question_counter:03d}"

            questions.append(
                Question(
                    question_id=question_id,
                    question_type="temporal: single cue retrieval",
                    subcategory="event sequence (location-based)",
                    cues="L",
                    target="Temporal",
                    question=question_text,
                    options=mc_options,
                    answer=answer_letter,
                    answer_index=answer_index,
                    related_events=[
                        labeled_ids[f"event_{i+1}"] for i in range(len(chosen_events))
                    ],
                    episode_span=[episode_id],
                    clip_span=list(
                        set(
                            e["scene_id"].replace(f"{episode_id}_", "")
                            for e in chosen_events
                        )
                    ),
                )
            )

        return questions


async def _process_single_episode(
    episode_id: str,
    episode_data: Dict[str, Any],
    events_by_episode: Dict[str, List[Event]],
    question_generator: QuestionGenerator,
    output_dir: Path,
    max_episode_span: int,
    temporal_questions_per_span: int,
    semaphore: asyncio.Semaphore,
) -> tuple:
    """
    Process a single episode asynchronously with proper resource management.

    Generates questions, calculates statistics, creates QA dataset, and saves to file.
    Uses semaphore for concurrency control to prevent resource exhaustion.

    Args:
        episode_id: Episode identifier
        episode_data: Episode data dictionary with scenes and events
        events_by_episode: Dictionary mapping episode_id to events for temporal reasoning
        question_generator: QuestionGenerator instance
        output_dir: Output directory for QA files
        max_episode_span: Maximum number of episodes to span for temporal questions
        temporal_questions_per_span: Number of temporal questions per span
        semaphore: Semaphore to control concurrent processing

    Returns:
        Tuple of (success: bool, episode_id: str, num_questions: int, error_message: Optional[str])
    """
    try:
        async with semaphore:
            questions = question_generator.generate_all_questions(
                episode_data,
                events_by_episode,
                episode_id,
                max_episode_span,
                temporal_questions_per_span,
            )

            stats = calculate_statistics(questions)
            print_statistics_report(stats, episode_id)

            qa_dataset = {
                "episode": episode_id,
                "title": episode_data.get("title", ""),
                "qa_dataset": [asdict(q) for q in questions],
                "statistics": stats,
            }

            output_filename = f"{episode_id}_qa.json"
            output_path = output_dir / output_filename
            save_json(qa_dataset, str(output_path))

        return True, episode_id, len(questions), None

    except Exception as e:
        error_msg = f"Error processing episode {episode_id}: {e}"
        return False, episode_id, 0, error_msg


async def _async_process_directory(
    input_dir: Path,
    output_dir: Path,
    question_generator: QuestionGenerator,
    max_episode_span: int,
    temporal_questions_per_span: int,
    max_files: Optional[int],
    max_concurrent_files: int,
) -> Dict[str, int]:
    """
    Async implementation of directory processing with parallel episode handling.

    Loads all episodes first for temporal reasoning across episodes, then processes
    episodes in parallel with progress tracking. Uses semaphore to limit concurrent
    processing and prevent resource exhaustion.

    Args:
        input_dir: Input directory containing JSON files with events
        output_dir: Output directory for QA JSON files
        question_generator: QuestionGenerator instance
        max_episode_span: Maximum number of episodes to span
        temporal_questions_per_span: Number of temporal questions per span
        max_files: Maximum number of files to process, None for all
        max_concurrent_files: Maximum number of concurrent file operations

    Returns:
        Dictionary with statistics: total_files, processed, failed, total_questions
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if max_files:
        json_files = json_files[:max_files]

    print(f"\n{'='*70}")
    print(f"Processing {len(json_files)} files from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max concurrent files: {max_concurrent_files}")
    print(f"{'='*70}\n")

    episodes_data = {}
    events_by_episode = {}

    print("Loading episode data...")
    for json_file in tqdm(json_files, desc="Loading files"):
        episode_data = load_episode_data(str(json_file))
        episode_id = episode_data["episode"]
        episodes_data[episode_id] = episode_data

        events = []
        for scene in episode_data["scenes"]:
            for e in scene.get("events", []):
                events.append(Event(**e) if isinstance(e, dict) else e)
        events_by_episode[episode_id] = events

    print(f"Loaded {len(episodes_data)} episodes\n")

    semaphore = asyncio.Semaphore(max_concurrent_files)

    tasks = [
        _process_single_episode(
            episode_id,
            episode_data,
            events_by_episode,
            question_generator,
            output_dir,
            max_episode_span,
            temporal_questions_per_span,
            semaphore,
        )
        for episode_id, episode_data in episodes_data.items()
    ]

    stats_dict = {
        "total_files": len(episodes_data),
        "processed": 0,
        "failed": 0,
        "total_questions": 0,
    }
    results = []

    print("Generating questions...")
    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Processing episodes"):
        success, episode_id, num_questions, error_msg = await coro
        if success:
            stats_dict["processed"] += 1
            stats_dict["total_questions"] += num_questions
        else:
            stats_dict["failed"] += 1
            if error_msg:
                print(f"\n{error_msg}")
        results.append((success, episode_id, num_questions, error_msg))

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total episodes:        {stats_dict['total_files']}")
    print(f"Processed:             {stats_dict['processed']}")
    print(f"Failed:                {stats_dict['failed']}")
    print(f"Total questions:       {stats_dict['total_questions']}")
    print(f"Output directory:      {output_dir}")
    print(f"{'='*70}\n")

    return stats_dict


def generate_qa_from_events(
    input_dir: Path,
    output_dir: Path,
    max_episode_span: int = 3,
    temporal_questions_per_span: int = 2,
    max_files: Optional[int] = None,
    max_concurrent_files: int = 5,
    batch_id: Optional[str] = None,
    metadata_file: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Generate QA datasets from episode JSON files that already have events.

    Main entry point for QA generation. Optionally downloads and processes batch results first,
    then processes all JSON files in input_dir containing episodes with events, generates
    multiple question types, and saves QA datasets to output_dir. Supports parallel processing
    for efficiency.

    Args:
        input_dir: Input directory path containing JSON files with events
        output_dir: Output directory path for QA JSON files
        max_episode_span: Maximum number of episodes to span for temporal questions (default: 3)
        temporal_questions_per_span: Number of temporal questions per span range (default: 2)
        max_files: Maximum number of files to process, None for all (default: None)
        max_concurrent_files: Maximum number of files to process concurrently (default: 5)
        batch_id: Optional OpenAI batch job ID to download and process results first (default: None)
        metadata_file: Optional metadata file path from batch submission (required if batch_id is provided)

    Returns:
        Dictionary containing processing statistics:
            - total_files: Total number of files found
            - processed: Number of successfully processed files
            - failed: Number of failed files
            - total_questions: Total number of questions generated
            - If batch processing: also includes total_scenes, episodes_created from batch results
    """
    combined_stats = {}

    if batch_id:
        if not metadata_file:
            raise ValueError(
                "metadata_file must be provided when batch_id is specified"
            )

        print(f"\n{'='*70}")
        print("STEP 1: DOWNLOADING AND PROCESSING BATCH RESULTS")
        print(f"{'='*70}\n")

        batch_stats = download_and_process_batch_results(
            batch_id=batch_id,
            metadata_file=metadata_file,
            output_dir=input_dir,  # Output batch results to input_dir for QA processing
        )
        combined_stats.update(batch_stats)

        print(f"\n{'='*70}")
        print("STEP 2: GENERATING QA DATASETS")
        print(f"{'='*70}\n")

    question_generator = QuestionGenerator()

    qa_stats = asyncio.run(
        _async_process_directory(
            input_dir,
            output_dir,
            question_generator,
            max_episode_span,
            temporal_questions_per_span,
            max_files,
            max_concurrent_files,
        )
    )

    combined_stats.update(qa_stats)
    return combined_stats
