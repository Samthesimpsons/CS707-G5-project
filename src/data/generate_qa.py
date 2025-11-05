#!/usr/bin/env python3
"""
Question Generation from Enhanced Episode Data

This script generates QA datasets from episode JSON files that already have events.
Use this after running add_events.py to separate event extraction from question generation.

Usage:
    python generate_qa_from_events.py --input_dir ./data_with_events --output_dir ./qa_output
    python generate_qa_from_events.py --input_file 0102_with_events.json --output_file 0102_qa.json
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import random


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
    """Represents a generated question."""

    question_id: str
    question_type: str
    subcategory: str
    cues: str = ""
    target: str = ""
    question: str = ""
    options: List[str] = None
    answer: str = ""
    answer_index: int = -1
    explanation: str = ""
    related_events: List[str] = None
    episode_span: List[str] = None
    clip_span: List[str] = None

    def __post_init__(self):
        if self.options is None:
            self.options = []
        if self.related_events is None:
            self.related_events = []
        if self.episode_span is None:
            self.episode_span = []
        if self.clip_span is None:
            self.clip_span = []


class QuestionGenerator:
    """Generates questions from events."""

    def __init__(self):
        """Initialize question generator."""
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
            events_by_episode: Dictionary mapping episode_id to events (for temporal questions)
            current_episode_id: Current episode being processed
            max_episode_span: Maximum number of episodes to span for temporal questions (default: 3)
            temporal_questions_per_span: Number of temporal questions to generate per span range (default: 2)

        Returns:
            List of Question objects
        """
        questions = []
        episode_id = episode_data["episode"]

        # Get events for current episode
        events = []
        for scene in episode_data["scenes"]:
            scene_events = scene.get("events", [])
            # Convert to Event objects if they're dicts
            for e in scene_events:
                if isinstance(e, dict):
                    events.append(Event(**e))
                else:
                    events.append(e)

        if not events:
            print(f"  ⚠ Warning: No events found in episode {episode_id}")
            return questions

        # Generate single target recall questions (60%)
        str_questions = self._generate_single_target_recall(events, episode_id)
        questions.extend(str_questions)

        # Generate boolean questions (20%)
        bool_questions = self._generate_boolean_questions(events, episode_id)
        questions.extend(bool_questions)

        # Generate temporal ordering questions (15%) - default 2 questions per span
        temporal_general_questions = self._generate_temporal_ordering(
            events_by_episode,
            current_episode_id,
            max_episode_span,
            temporal_questions_per_span,
        )
        questions.extend(temporal_general_questions)

        # Generate latest event retrieval questions (5%)
        temporal_latest_questions = self._generate_latest_event_retrieval(
            events, episode_id
        )
        questions.extend(temporal_latest_questions)

        # Generate temporal single cue retrieval questions (location-based ordering) - 2 questions per episode
        temporal_location_questions = self._generate_temporal_single_cue_retrieval(
            events, current_episode_id, num_questions=2, num_events=4
        )
        questions.extend(temporal_location_questions)

        return questions

    def _generate_single_target_recall(
        self, events: List[Event], episode_id: str
    ) -> List[Question]:
        """Generate single target recall questions (E+S→L, L+E→S, L+S→E)."""
        questions = []

        # Get all unique locations
        all_locations = list(set(e.location for e in events if e.location))

        for event in events:
            if not event.location:
                continue

            # E+S → L (Location recall)
            q = self._create_location_recall_question(event, episode_id, all_locations)
            if q:
                questions.append(q)

            # L+E → S (Subject recall)
            q = self._create_subject_recall_question(event, episode_id, events)
            if q:
                questions.append(q)

            # L+S → E (Event recall)
            q = self._create_event_recall_question(event, episode_id, events)
            if q:
                questions.append(q)

        return questions

    def _create_location_recall_question(
        self, event: Event, episode_id: str, all_locations: List[str]
    ) -> Optional[Question]:
        """Create E+S→L question."""
        if len(event.involved_subjects) == 0:
            return None

        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"

        subjects_str = " and ".join(event.involved_subjects)
        question_text = f"Where did {event.event_description.lower()} when {subjects_str} were present?"

        # Generate distractors
        distractors = [loc for loc in all_locations if loc != event.location]
        options_raw = [event.location] + random.sample(
            distractors, min(3, len(distractors))
        )

        # Ensure we have 4 options
        while len(options_raw) < 4:
            options_raw.append(f"Unknown location {len(options_raw)}")
        options_raw = options_raw[:4]

        # Randomly place correct answer at any position
        answer_index = random.randint(0, 3)
        options = [None] * 4
        options[answer_index] = event.location

        # Fill remaining positions with distractors
        distractor_idx = 0
        for i in range(4):
            if options[i] is None:
                options[i] = options_raw[1 + distractor_idx]
                distractor_idx += 1

        answer_letter = chr(65 + answer_index)  # A, B, C, D

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
        """Create L+E→S question."""
        if len(event.involved_subjects) == 0:
            return None

        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"

        question_text = f"Who was present at {event.location} when {event.event_description.lower()}?"

        # Generate distractors - other subject combinations from same location
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

        # Add some variations
        if len(distractors) < 3 and len(event.involved_subjects) > 1:
            distractors.append(", ".join(event.involved_subjects[:-1]))

        options_raw = [correct_answer] + distractors[:3]

        # Ensure 4 options
        while len(options_raw) < 4:
            options_raw.append(f"Unknown subjects {len(options_raw)}")
        options_raw = options_raw[:4]

        # Randomly place correct answer at any position
        answer_index = random.randint(0, 3)
        options = [None] * 4
        options[answer_index] = correct_answer

        distractor_idx = 0
        for i in range(4):
            if options[i] is None:
                options[i] = (
                    options_raw[1 + distractor_idx]
                    if 1 + distractor_idx < len(options_raw)
                    else f"Unknown subjects {i}"
                )
                distractor_idx += 1

        answer_letter = chr(65 + answer_index)  # A, B, C, D

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
        """Create L+S→E question."""
        if len(event.involved_subjects) == 0:
            return None

        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"

        subjects_str = " and ".join(event.involved_subjects)
        question_text = (
            f"What happened at {event.location} when {subjects_str} were together?"
        )

        # Generate distractors - other events at same location
        same_location_events = [
            e
            for e in all_events
            if e.location == event.location and e.event_id != event.event_id
        ]

        distractors = [e.event_description for e in same_location_events[:3]]

        options_raw = [event.event_description] + distractors

        # Ensure 4 options
        while len(options_raw) < 4:
            options_raw.append(f"Unknown event {len(options_raw)}")
        options_raw = options_raw[:4]

        # Randomly place correct answer at any position
        answer_index = random.randint(0, 3)
        options = [None] * 4
        options[answer_index] = event.event_description

        distractor_idx = 0
        for i in range(4):
            if options[i] is None:
                options[i] = (
                    options_raw[1 + distractor_idx]
                    if 1 + distractor_idx < len(options_raw)
                    else f"Unknown event {i}"
                )
                distractor_idx += 1

        answer_letter = chr(65 + answer_index)  # A, B, C, D

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
        """Generate boolean verification questions with negative examples."""
        questions = []

        all_locations = list(set(e.location for e in events if e.location))
        all_subjects = list(set(s for e in events for s in e.involved_subjects))

        # Sample events for boolean questions
        sampled_events = random.sample(
            events, min(len(events), max(1, len(events) // 5))
        )

        for event in sampled_events:
            # Location verification (false)
            wrong_locations = [loc for loc in all_locations if loc != event.location]
            if wrong_locations:
                self.question_counter += 1
                question_id = f"Q_{episode_id}_{self.question_counter:03d}"

                wrong_loc = random.choice(wrong_locations)
                question_text = f"Did {event.event_description.lower()} at {wrong_loc}?"

                # Randomly place "No" at position 0 or 1
                answer_index = random.randint(0, 1)
                options = ["Yes", "No"] if answer_index == 1 else ["No", "Yes"]
                answer_letter = chr(65 + options.index("No"))  # A or B

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

            # Subject verification (false)
            absent_subjects = [
                s for s in all_subjects if s not in event.involved_subjects
            ]
            if absent_subjects:
                self.question_counter += 1
                question_id = f"Q_{episode_id}_{self.question_counter:03d}"

                wrong_subject = random.choice(absent_subjects)
                question_text = f"Was {wrong_subject} present when {event.event_description.lower()}?"

                # Randomly place "No" at position 0 or 1
                answer_index = random.randint(0, 1)
                options = ["Yes", "No"] if answer_index == 1 else ["No", "Yes"]
                answer_letter = chr(65 + options.index("No"))  # A or B

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

        Args:
            events_by_episode: Dictionary mapping episode_id to events
            current_episode_id: Current episode being processed
            max_episode_span: Maximum number of episodes to span (default: 3)
            temporal_questions_per_span: Number of questions to generate per span range (default: 2)

        Returns:
            List of temporal ordering questions
        """
        questions = []

        # Get current episode events
        current_events = events_by_episode.get(current_episode_id, [])
        if len(current_events) < 2:
            return questions

        # Get episode IDs sorted
        episode_ids = sorted(events_by_episode.keys())
        current_idx = (
            episode_ids.index(current_episode_id)
            if current_episode_id in episode_ids
            else 0
        )

        # Generate questions for different episode spans (1 to max_episode_span)
        for span in range(1, max_episode_span + 1):
            # Get relevant episode IDs for this span
            start_idx = max(0, current_idx - span + 1)
            end_idx = min(len(episode_ids), current_idx + 1)
            relevant_episode_ids = episode_ids[start_idx:end_idx]

            if len(relevant_episode_ids) < span:
                continue  # Not enough episodes for this span

            # Collect events from relevant episodes
            span_events = []
            for ep_id in relevant_episode_ids:
                span_events.extend(events_by_episode.get(ep_id, []))

            # Sort events by event_id (assumes chronological ordering)
            sorted_events = sorted(span_events, key=lambda e: e.event_id)

            if len(sorted_events) < 2:
                continue

            # Generate multiple questions per span configuration
            num_questions = min(temporal_questions_per_span, len(sorted_events) - 1)

            for q_idx in range(num_questions):
                # Select event pairs with some distance between them
                # Distribute pairs across the entire span to get diverse questions
                if len(sorted_events) > 3:
                    # Calculate step size to distribute questions evenly
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

                # Generate distractors - events from different time points
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

                # Randomly place correct answer at any position
                answer_index = random.randint(0, 3)
                options = [None] * 4
                options[answer_index] = event_a.event_description

                distractor_idx = 0
                for k in range(4):
                    if options[k] is None:
                        options[k] = (
                            options_raw[1 + distractor_idx]
                            if 1 + distractor_idx < len(options_raw)
                            else f"Unknown event {k}"
                        )
                        distractor_idx += 1

                answer_letter = chr(65 + answer_index)  # A, B, C, D

                # Determine episode span
                episode_span = list(
                    set(
                        [
                            event_a.event_id.split("_scene_")[0],
                            event_b.event_id.split("_scene_")[0],
                        ]
                    )
                )

                # Determine clip span
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
        """Generate latest event retrieval questions."""
        questions = []

        if not events:
            return questions

        # Group events by location
        events_by_location = defaultdict(list)
        for event in events:
            if event.location:
                events_by_location[event.location].append(event)

        # Latest event at each location
        for location, loc_events in events_by_location.items():
            if len(loc_events) < 2:
                continue

            # Assuming last event in list is the latest
            latest_event = loc_events[-1]

            self.question_counter += 1
            question_id = f"Q_{episode_id}_{self.question_counter:03d}"

            question_text = (
                f"What was the last event that happened at {location} in this episode?"
            )

            # Distractors are earlier events at same location
            distractors = [e.event_description for e in loc_events[:-1]][-3:]

            options_raw = [latest_event.event_description] + distractors

            while len(options_raw) < 4:
                options_raw.append(f"Unknown event {len(options_raw)}")
            options_raw = options_raw[:4]

            # Randomly place correct answer at any position
            answer_index = random.randint(0, 3)
            options = [None] * 4
            options[answer_index] = latest_event.event_description

            distractor_idx = 0
            for i in range(4):
                if options[i] is None:
                    options[i] = (
                        options_raw[1 + distractor_idx]
                        if 1 + distractor_idx < len(options_raw)
                        else f"Unknown event {i}"
                    )
                    distractor_idx += 1

            answer_letter = chr(65 + answer_index)  # A, B, C, D

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

            break  # Only one location-based question per episode

        # Latest event in entire episode
        if len(events) >= 2:
            latest_event = events[-1]

            self.question_counter += 1
            question_id = f"Q_{episode_id}_{self.question_counter:03d}"

            question_text = (
                "What was the most recent event that occurred in the episode?"
            )

            # Distractors from earlier events
            distractors = [e.event_description for e in events[:-1]][-3:]

            options_raw = [latest_event.event_description] + distractors

            while len(options_raw) < 4:
                options_raw.append(f"Unknown event {len(options_raw)}")
            options_raw = options_raw[:4]

            # Randomly place correct answer at any position
            answer_index = random.randint(0, 3)
            options = [None] * 4
            options[answer_index] = latest_event.event_description

            distractor_idx = 0
            for i in range(4):
                if options[i] is None:
                    options[i] = (
                        options_raw[1 + distractor_idx]
                        if 1 + distractor_idx < len(options_raw)
                        else f"Unknown event {i}"
                    )
                    distractor_idx += 1

            answer_letter = chr(65 + answer_index)  # A, B, C, D

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
        Generate temporal single cue retrieval questions (ordering events by location).
        This asks users to order events that occurred at the same location chronologically.

        Args:
            events: List of events from the current episode
            episode_id: Current episode ID
            num_questions: Number of questions to generate (default: 2)
            num_events: Number of events per question (default: 4)

        Returns:
            List of temporal single cue retrieval questions
        """
        questions = []

        if not events:
            return questions

        # Convert Event objects to dict format for processing
        events_data = []
        for event in events:
            # Extract scene number from event_id
            parts = event.event_id.split("_")
            scene_num = parts[2] if len(parts) > 2 else "000"

            # Parse timestamp to get start seconds
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

        # Group events by location
        locations = defaultdict(list)
        for ev in events_data:
            if ev["location"]:
                locations[ev["location"]].append(ev)

        # Find valid locations (appear in multiple scenes with enough events)
        valid_locations = [
            loc
            for loc, evs in locations.items()
            if len(set(e["scene_id"] for e in evs)) > 1 and len(evs) >= num_events
        ]

        if not valid_locations:
            # If no multi-scene locations, try single-scene locations
            valid_locations = [
                loc for loc, evs in locations.items() if len(evs) >= num_events
            ]

        if not valid_locations:
            return questions

        # Generate questions
        for q_idx in range(min(num_questions, len(valid_locations))):
            location = (
                valid_locations[q_idx]
                if q_idx < len(valid_locations)
                else random.choice(valid_locations)
            )
            loc_events = locations[location]

            # Sample events, preferring diversity across scenes
            scenes = list({e["scene_id"] for e in loc_events})
            random.shuffle(scenes)
            chosen_events = []
            used_scenes = set()

            # First pass: one event per scene
            for e in random.sample(loc_events, min(len(loc_events), num_events * 2)):
                if e["scene_id"] not in used_scenes:
                    chosen_events.append(e)
                    used_scenes.add(e["scene_id"])
                if len(chosen_events) == num_events:
                    break

            # Second pass: fill remaining slots
            if len(chosen_events) < num_events:
                remaining = [e for e in loc_events if e not in chosen_events]
                chosen_events += random.sample(
                    remaining, min(len(remaining), num_events - len(chosen_events))
                )

            if len(chosen_events) < 2:
                continue

            # Take only num_events
            chosen_events = chosen_events[:num_events]

            # Create labeled events
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

            # Correct sequence based on start_seconds
            correct_sequence = [
                label
                for label, _ in sorted(labeled_start.items(), key=lambda kv: kv[1])
            ]
            all_labels = list(labeled.keys())

            # Generate options
            options_sequences = [correct_sequence.copy()]
            seen = {tuple(correct_sequence)}

            # Generate 3 incorrect permutations
            max_attempts = 50
            attempts = 0
            while len(options_sequences) < 4 and attempts < max_attempts:
                perm = random.sample(all_labels, len(all_labels))
                if tuple(perm) not in seen:
                    options_sequences.append(perm)
                    seen.add(tuple(perm))
                attempts += 1

            # If we couldn't generate enough permutations, create some manually
            while len(options_sequences) < 4:
                # Swap two adjacent elements
                perm = correct_sequence.copy()
                if len(perm) >= 2:
                    idx = random.randint(0, len(perm) - 2)
                    perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
                    if tuple(perm) not in seen:
                        options_sequences.append(perm)
                        seen.add(tuple(perm))
                    else:
                        options_sequences.append(perm)  # Add anyway if we're stuck
                        break

            # Randomly place correct answer at any position
            answer_index = random.randint(0, 3)

            # Rearrange options to place correct answer at answer_index
            final_options = [None] * 4
            final_options[answer_index] = correct_sequence

            opt_idx = 0
            for i in range(4):
                if final_options[i] is None:
                    if opt_idx + 1 < len(options_sequences):
                        final_options[i] = options_sequences[opt_idx + 1]
                        opt_idx += 1
                    else:
                        # Create a different permutation
                        perm = correct_sequence.copy()
                        random.shuffle(perm)
                        final_options[i] = perm

            # Format options as strings
            letters = ["A", "B", "C", "D"]
            mc_options = [
                f"{letters[i]}. {', '.join(final_options[i])}" for i in range(4)
            ]
            answer_letter = letters[answer_index]

            # Create question text
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


##================= Statistics Calculation and Reporting =================##
def calculate_statistics(questions: List[Question]) -> Dict[str, Any]:
    """Calculate statistics for generated questions."""
    stats = {
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

        # Track episode span
        if q.episode_span:
            span_count = len(q.episode_span)
            stats["episode_span_distribution"][f"{span_count} episode(s)"] += 1

        # Track clip span
        if q.clip_span:
            clip_count = len(q.clip_span)
            stats["clip_span_distribution"][f"{clip_count} clip(s)"] += 1

        # Track answer distribution
        if isinstance(q.answer, str) and len(q.answer) == 1 and q.answer in "ABCD":
            stats["answer_distribution"][q.answer] += 1
        elif q.answer_index >= 0 and q.answer_index < 4:
            answer_letter = chr(65 + q.answer_index)
            stats["answer_distribution"][answer_letter] += 1

    # Convert defaultdict to regular dict
    return {
        "total_questions": stats["total_questions"],
        "question_type_distribution": dict(stats["question_type_distribution"]),
        "subcategory_distribution": dict(stats["subcategory_distribution"]),
        "cue_distribution": dict(stats["cue_distribution"]),
        "target_distribution": dict(stats["target_distribution"]),
        "episode_span_distribution": dict(stats["episode_span_distribution"]),
        "clip_span_distribution": dict(stats["clip_span_distribution"]),
        "answer_distribution": dict(stats["answer_distribution"]),
    }


def print_statistics_report(stats: Dict[str, Any], episode_id: str = ""):
    """
    Print a formatted statistics report.

    Args:
        stats: Statistics dictionary from calculate_statistics
        episode_id: Episode ID for the report header
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


def load_episode_data(json_path: str) -> Dict[str, Any]:
    """Load episode data from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, output_path: str):
    """Save data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA dataset from enhanced episode data with events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate QA from single file
  python generate_qa_from_events.py \\
      --input_file 0102_with_events.json \\
      --output_file 0102_qa.json
  
  # Process entire directory
  python generate_qa_from_events.py \\
      --input_dir ./data_with_events \\
      --output_dir ./qa_output
  
  # Custom episode span (up to 5 episodes)
  python generate_qa_from_events.py \\
      --input_dir ./data_with_events \\
      --output_dir ./qa_output \\
      --max_episode_span 5
  
  # Generate multiple temporal questions per span (3 questions for each span range)
  python generate_qa_from_events.py \\
      --input_dir ./data_with_events \\
      --output_dir ./qa_output \\
      --temporal_questions_per_span 3
  
  # Combine both: 5-episode span with 2 questions per span
  python generate_qa_from_events.py \\
      --input_dir ./data_with_events \\
      --output_dir ./qa_output \\
      --max_episode_span 5 \\
      --temporal_questions_per_span 2
        """,
    )

    parser.add_argument(
        "--input_file", type=str, help="Single enhanced episode JSON file"
    )
    parser.add_argument("--output_file", type=str, help="Output QA file path")
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing enhanced episode files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qa_output",
        help="Directory to save QA files (default: %(default)s)",
    )
    parser.add_argument(
        "--max_files", type=int, default=None, help="Maximum number of files to process"
    )
    parser.add_argument(
        "--max_episode_span",
        type=int,
        default=3,
        help="Maximum number of episodes to span for temporal questions (default: 3)",
    )
    parser.add_argument(
        "--temporal_questions_per_span",
        type=int,
        default=2,
        help="Number of temporal questions to generate per span range (default: 2)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input_file and not args.input_dir:
        parser.error("Either --input_file or --input_dir must be specified")

    if args.input_file and not args.output_file:
        parser.error("--output_file required when using --input_file")

    # Initialize generator
    question_generator = QuestionGenerator()

    # Single file mode
    if args.input_file:
        print(f"\nGenerating QA for: {args.input_file}")

        episode_data = load_episode_data(args.input_file)
        episode_id = episode_data["episode"]

        # Get events
        events_by_episode = {episode_id: []}
        for scene in episode_data["scenes"]:
            for e in scene.get("events", []):
                events_by_episode[episode_id].append(
                    Event(**e) if isinstance(e, dict) else e
                )

        # Generate questions
        questions = question_generator.generate_all_questions(
            episode_data,
            events_by_episode,
            episode_id,
            args.max_episode_span,
            args.temporal_questions_per_span,
        )

        print(f"  Generated {len(questions)} questions")

        # Calculate statistics
        stats = calculate_statistics(questions)

        # Print statistics report
        print_statistics_report(stats, episode_id)

        # Create QA dataset
        qa_dataset = {
            "episode": episode_id,
            "title": episode_data.get("title", ""),
            "qa_dataset": [asdict(q) for q in questions],
            "statistics": stats,
        }

        # Save
        save_json(qa_dataset, args.output_file)
        print(f"✓ Saved: {args.output_file}\n")

        return 0

    # Directory mode
    os.makedirs(args.output_dir, exist_ok=True)

    input_path = Path(args.input_dir)
    json_files = sorted(input_path.glob("*.json"))

    if args.max_files:
        json_files = json_files[: args.max_files]

    print(f"\nProcessing {len(json_files)} files")
    print(f"Output directory: {args.output_dir}\n")

    # Load all episodes for temporal reasoning
    episodes_data = {}
    events_by_episode = {}

    print("Loading episode data...")
    for json_file in json_files:
        episode_data = load_episode_data(str(json_file))
        episode_id = episode_data["episode"]
        episodes_data[episode_id] = episode_data

        # Extract events
        events = []
        for scene in episode_data["scenes"]:
            for e in scene.get("events", []):
                events.append(Event(**e) if isinstance(e, dict) else e)
        events_by_episode[episode_id] = events

    print(f"Loaded {len(episodes_data)} episodes\n")

    # Track overall statistics
    all_questions = []

    # Generate QA for each episode
    for episode_id, episode_data in episodes_data.items():
        print(f"Generating QA for episode {episode_id}...")

        questions = question_generator.generate_all_questions(
            episode_data,
            events_by_episode,
            episode_id,
            args.max_episode_span,
            args.temporal_questions_per_span,
        )

        all_questions.extend(questions)
        print(f"  Generated {len(questions)} questions")

        # Calculate statistics
        stats = calculate_statistics(questions)

        # Print statistics report for this episode
        print_statistics_report(stats, episode_id)

        # Create QA dataset
        qa_dataset = {
            "episode": episode_id,
            "title": episode_data.get("title", ""),
            "qa_dataset": [asdict(q) for q in questions],
            "statistics": stats,
        }

        # Save
        output_filename = f"{episode_id}_qa.json"
        output_path = os.path.join(args.output_dir, output_filename)
        save_json(qa_dataset, output_path)
        print(f"✓ Saved: {output_path}\n")

    # Print overall statistics summary
    if len(episodes_data) > 1:
        overall_stats = calculate_statistics(all_questions)
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS ACROSS ALL EPISODES")
        print("=" * 70)
        print(f"\nTotal Episodes Processed: {len(episodes_data)}")
        print_statistics_report(overall_stats, "")

    print(f"✓ Complete! Generated QA for {len(episodes_data)} episodes")
    print(f"Output directory: {args.output_dir}\n")

    return 0


if __name__ == "__main__":
    exit(main())
