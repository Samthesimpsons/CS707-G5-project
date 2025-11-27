import json
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from dotenv import load_dotenv
from litellm import completion
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialogueTurn(TypedDict):
    """Single line of dialogue in a scene.

    Attributes:
        speaker: Normalized speaker name.
        text: Dialogue text spoken by the speaker.
    """

    speaker: str
    text: str


class Scene(TypedDict, total=False):
    """Scene metadata including dialogue, subjects, location, and timing.

    Attributes:
        scene_description: Raw scene heading or description.
        dialogue: Ordered dialogue turns within the scene.
        subjects: Unique speakers present in the scene.
        location: Parsed location for the scene.
        clip_start: Optional start time for the video clip.
        clip_end: Optional end time for the video clip.
        srt_start: Optional subtitle start time.
        srt_end: Optional subtitle end time.
        events: Extracted atomic events for the scene.
    """

    scene_description: str
    dialogue: list[DialogueTurn]
    subjects: list[str]
    location: str
    clip_start: str
    clip_end: str
    srt_start: str | int
    srt_end: str | int
    events: list[dict[str, object]]


class EpisodeData(TypedDict):
    """Structured representation of an episode.

    Attributes:
        episode: Episode identifier (e.g., 0101).
        title: Episode title.
        scenes: Ordered list of scenes with dialogue and annotations.
    """

    episode: str
    title: str
    scenes: list[Scene]


@dataclass
class ProcessConfig:
    """Configuration for the end-to-end event extraction and QA pipeline.

    Attributes:
        input_dir: Directory of annotated episode JSON files.
        output_dir: Destination for episode files enriched with events.
        model_path: Path to the local model for LLM extraction.
        llm_backend: Backend to use for event extraction (`openai` or `local`).
        openai_model: Model name to use when backend is `openai`.
        qa_output_dir: Directory to store generated QA datasets.
        device: Device mapping passed to the transformer model.
        max_events: Max events to extract per scene.
        max_files: Optional cap on how many files to process.
        overwrite: Whether to replace existing outputs.
        resume: Whether to skip already processed files.
        max_episode_span: Max number of episodes spanned for temporal questions.
        temporal_questions_per_span: Number of temporal questions per span size.
        temperature: Sampling temperature for LLM outputs.
    """

    input_dir: Path = Path("./data/annotated_tuples")
    output_dir: Path = Path("./data/output_with_events")
    model_path: str = "./models/DeepSeek-R1-Distill-Qwen-7B"
    llm_backend: str = "openai"  # "local"
    openai_model: str = "gpt-4.1-nano-2025-04-14"
    qa_output_dir: Path = Path("./data/qa_output")
    device: str = "auto"
    max_events: int = 10
    max_files: int | None = None
    overwrite: bool = False
    resume: bool = True
    max_episode_span: int = 3
    temporal_questions_per_span: int = 2
    temperature: float = 1.0  # 0.3


@dataclass
class Event:
    """Represents an atomic event in the narrative.

    Attributes:
        event_id: Unique event identifier.
        event_description: Human-readable description of the event.
        involved_subjects: Characters actively participating in the event.
        location: Location where the event occurs.
        timestamp: Start-end timestamp window for the event.
        episode_clip: Clip identifier linking back to the episode/scene.
    """

    event_id: str
    event_description: str
    involved_subjects: list[str]
    location: str
    timestamp: str
    episode_clip: str = ""


class LLMEventExtractor:
    """Extracts atomic events from scenes using a local or hosted LLM."""

    def __init__(
        self,
        model_path: str,
        temperature: float,
        device: str = "auto",
        backend: str = "local",
        openai_model: str | None = None,
    ) -> None:
        """Initialize the LLM model and tokenizer.

        Args:
            model_path: Local model path used when backend is `local`.
            temperature: Sampling temperature for text generation.
            device: Device placement hint for transformer loading.
            backend: LLM backend selection (`local` or `openai`).
            openai_model: Optional OpenAI model name when backend is `openai`.

        Raises:
            EnvironmentError: If `OPENAI_API_KEY` is required but missing.
            RuntimeError: If model loading fails for the local backend.
        """
        self.backend = backend
        self.openai_model = openai_model or "gpt-4.1-nano-2025-04-14"
        self.temperature = temperature
        self.model: Any = None
        self.tokenizer: Any = None
        if self.backend == "openai":
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Add it to your environment or .env file."
                )
            self.model = None
            self.tokenizer = None
            print(f"Using OpenAI model {self.openai_model} via LiteLLM.")
            return

        print(f"Loading model from {model_path}...")
        tokenizer_cls = AutoTokenizer
        model_cls = AutoModelForCausalLM
        self.tokenizer = tokenizer_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded successfully.")

    def extract_events(
        self,
        scene: Scene,
        episode_id: str,
        scene_idx: int,
        max_events: int = 10,
    ) -> list[Event]:
        """Extract atomic events from a single scene.

        Args:
            scene: Scene data containing dialogue, subjects, and metadata.
            episode_id: Episode identifier the scene belongs to.
            scene_idx: Zero-padded scene index within the episode.
            max_events: Maximum number of events to return.

        Returns:
            List of `Event` instances extracted from the scene.
        """
        prompt = self._create_extraction_prompt(
            scene=scene,
            episode_id=episode_id,
            scene_id=f"{scene_idx:03d}",
            max_events=max_events,
        )
        response = self._generate_response(prompt, max_tokens=1500)
        events = self._parse_events_from_response(
            response, scene, episode_id, scene_idx
        )
        return events[:max_events]

    def _format_dialogue(self, scene: Scene) -> str:
        """Return scene dialogue as newline-separated `SPEAKER: text` lines.

        Args:
            scene: Scene containing dialogue turns.

        Returns:
            Concatenated dialogue string for prompt construction.
        """
        dialogue = scene.get("dialogue", [])
        return "\n".join(
            f"{turn['speaker']}: {turn['text']}"
            for turn in dialogue
            if "speaker" in turn and "text" in turn
        )

    def _create_extraction_prompt(
        self,
        scene: Scene,
        episode_id: str,
        scene_id: str,
        max_events: int,
    ) -> str:
        """Create the prompt for event extraction.

        Args:
            scene: Scene data with dialogue and metadata.
            episode_id: Episode identifier.
            scene_id: Zero-padded scene index string.
            max_events: Maximum number of events to request.

        Returns:
            Prompt string sent to the LLM.
        """
        dialogue_text = self._format_dialogue(scene)
        location = scene.get("location", "Unknown")
        subjects = ", ".join(scene.get("subjects", []))
        clip_range = f"{scene.get('clip_start', '')} to {scene.get('clip_end', '')}"
        scene_description = scene.get("scene_description", "N/A")

        prompt = f"""You are an expert at analyzing narrative scenes and extracting atomic events.

Scene Information:
Location: {location}
Characters Present: {subjects}
Duration: {clip_range}
Scene Description: {scene_description}

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
- Involved subjects should be characters who are present in the environment when the event occurs
- Example: When Ross and Susan are in the room, and Ross talks to himself, as Susan was there, both Ross and Susan should be listed as involved subjects.
- If Susan left the room before Ross talks to himself, only Ross should be listed as an involved subject.

Format your response as a JSON array (output ONLY the JSON, nothing else):
[
  {{
    "event_id": "{episode_id}_scene_{scene_id}_event_001",
    "event_description": "Clear, specific description of what happened",
    "involved_subjects": ["CHARACTER1", "CHARACTER2"],
    "location": "{location}",
    "timestamp": "00:00:05-00:00:20",
    "episode_clip": "{episode_id}_scene_{scene_id}"
  }},
  {{
    "event_id": "{episode_id}_scene_{scene_id}_event_002",
    "event_description": "Another event description",
    "involved_subjects": ["CHARACTER1"],
    "location": "{location}",
    "timestamp": "00:00:21-00:00:35",
    "episode_clip": "{episode_id}_scene_{scene_id}"
  }}
]

Important:
- event_id should follow the pattern: {episode_id}_scene_{scene_id}_event_XXX (where XXX is a 3-digit number starting from 001)
- timestamp should be in format "HH:MM:SS-HH:MM:SS" (start-end)
- location should be: {location}
- episode_clip should be: {episode_id}_scene_{scene_id}

Output only the JSON array, no additional text."""
        return prompt

    def _generate_response(self, prompt: str, max_tokens: int = 1500) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: Prompt text to send.
            max_tokens: Maximum tokens to generate.

        Returns:
            Model-generated text response (may be empty on error).

        Raises:
            RuntimeError: If local backend is selected without a loaded model.
        """
        if self.backend == "openai":
            messages = [{"role": "user", "content": prompt}]
            try:
                response = completion(
                    model=self.openai_model,
                    messages=messages,
                    temperature=self.temperature,
                )
            except Exception as exc:
                print(f"Error generating response with OpenAI via LiteLLM: {exc}")
                return ""

            choice = response.choices[0]
            message = choice.message
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", message)

            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(
                            part.get("text")
                            or part.get("content")
                            or part.get("value")
                            or ""
                        )
                    else:
                        parts.append(str(part))
                content = "".join(parts)

            if not isinstance(content, str):
                content = str(content)
            return content

        messages = [{"role": "user", "content": prompt}]
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Local LLM backend selected but model/tokenizer not initialized."
            )
        tokenizer = cast(Any, self.tokenizer)
        model = cast(Any, self.model)
        model_device = getattr(model, "device", None)
        if model_device is None:
            raise RuntimeError("Model device is not available on the loaded model.")
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        )
        return response

    def _parse_events_from_response(
        self,
        response: str,
        scene: Scene,
        episode_id: str,
        scene_idx: int,
    ) -> list[Event]:
        """Parse events from the LLM response.

        Args:
            response: Raw text returned by the LLM.
            scene: Scene metadata used for defaults and validation.
            episode_id: Episode identifier for naming events.
            scene_idx: Scene index to embed in event ids.

        Returns:
            List of parsed `Event` objects.
        """
        events: list[Event] = []
        try:
            cleaned = response.strip()
            if "```" in cleaned:
                cleaned = re.sub(r"^.*?```(?:json)?", "", cleaned, flags=re.S)
                cleaned = re.sub(r"```.*$", "", cleaned, flags=re.S)
            if "[" in cleaned and "]" in cleaned:
                cleaned = cleaned[cleaned.index("[") : cleaned.rindex("]") + 1]
            response = cleaned

            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            events_data: object
            if json_match:
                events_data = json.loads(json_match.group())
            else:
                try:
                    events_data = json.loads(response)
                except Exception:
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

                event_id = f"{episode_id}_scene_{scene_idx:03d}_event_{idx + 1:03d}"
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
                episode_clip = event_data.get(
                    "episode_clip", f"{episode_id}_scene_{scene_idx:03d}"
                )

                events.append(
                    Event(
                        event_id=event_id,
                        event_description=event_desc,
                        involved_subjects=involved_subjects,
                        location=scene.get("location", ""),
                        timestamp=timestamp,
                        episode_clip=episode_clip,
                    )
                )
        except json.JSONDecodeError as exc:
            print(f"Warning: JSON decode error for scene {scene_idx}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"Warning: Unexpected error parsing events for scene {scene_idx}: {exc}"
            )
        return events


@dataclass
class Question:
    """Represents a generated question.

    Attributes:
        question_id: Unique identifier for the question.
        question_type: High-level question category.
        subcategory: More specific question subtype.
        cues: Cues available to answer the question (e.g., E+S).
        target: Target label or modality being queried.
        question: Natural language question text.
        options: Multiple-choice options (if applicable).
        answer: Correct answer label or option letter.
        answer_index: Index of the correct option (0-based) when applicable.
        explanation: Optional rationale for the answer.
        related_events: Event ids referenced by the question.
        episode_span: Episodes spanned by the question.
        clip_span: Clip identifiers spanned by the question.
    """

    question_id: str
    question_type: str
    subcategory: str
    cues: str = ""
    target: str = ""
    question: str = ""
    options: list[str] = field(default_factory=list)
    answer: str = ""
    answer_index: int = -1
    explanation: str = ""
    related_events: list[str] = field(default_factory=list)
    episode_span: list[str] = field(default_factory=list)
    clip_span: list[str] = field(default_factory=list)


class QuestionGenerator:
    """Generates questions from events."""

    def __init__(self) -> None:
        """Initialize generator state."""
        self.question_counter = 0

    def _is_valid_option(self, option: str) -> bool:
        """Check whether an answer option is usable.

        Args:
            option: Option text.

        Returns:
            True if the option should be kept; otherwise False.
        """
        if not option:
            return False
        option_lower = option.lower().strip()
        if option_lower == "all":
            return False
        if option_lower.startswith("unknown"):
            return False
        return True

    def _has_valid_text(self, text: str) -> bool:
        """Validate free-text strings for questions/answers.

        Args:
            text: Input text to evaluate.

        Returns:
            True if the text should be kept; otherwise False.
        """
        if not text:
            return True
        text_lower = text.lower()
        if re.search(r"\ball\b", text_lower):
            return False
        if "unknown" in text_lower:
            return False
        return True

    def _validate_and_deduplicate_options(self, options: list[str]) -> list[str]:
        """Filter invalid options and drop duplicates while preserving order.

        Args:
            options: Raw option list.

        Returns:
            Cleaned list of options.
        """
        if not options:
            return []
        seen = set()
        validated: list[str] = []
        for opt in options:
            opt_normalized = opt.strip().lower()
            if self._is_valid_option(opt) and opt_normalized not in seen:
                validated.append(opt)
                seen.add(opt_normalized)
        return validated

    def generate_all_questions(
        self,
        episode_data: EpisodeData,
        events_by_episode: dict[str, list[Event]],
        current_episode_id: str,
        max_episode_span: int = 3,
        temporal_questions_per_span: int = 2,
    ) -> list[Question]:
        """Generate a suite of questions for an episode and nearby context.

        Args:
            episode_data: Episode data with scenes and events.
            events_by_episode: Mapping from episode id to its events.
            current_episode_id: Episode id being processed.
            max_episode_span: Max number of episodes to consider for temporal questions.
            temporal_questions_per_span: Temporal questions per span size.

        Returns:
            List of generated `Question` objects.
        """
        questions: list[Question] = []
        episode_id = episode_data["episode"]

        events: list[Event] = []
        for scene in episode_data.get("scenes", []):
            for event in scene.get("events", []):
                if isinstance(event, dict):
                    events.append(Event(**event))  # type: ignore[arg-type]
                else:
                    events.append(event)

        if not events:
            print(f"Warning: No events found in episode {episode_id}")
            return questions

        questions.extend(self._generate_single_target_recall(events, episode_id))
        questions.extend(self._generate_boolean_questions(events, episode_id))
        questions.extend(
            self._generate_temporal_ordering(
                events_by_episode,
                current_episode_id,
                max_episode_span,
                temporal_questions_per_span,
            )
        )
        questions.extend(self._generate_latest_event_retrieval(events, episode_id))
        questions.extend(
            self._generate_temporal_single_cue_retrieval(
                events, current_episode_id, num_questions=2, num_events=4
            )
        )
        return questions

    def _generate_single_target_recall(
        self, events: list[Event], episode_id: str
    ) -> list[Question]:
        """Create recall questions about location, subjects, and events.

        Args:
            events: Events extracted from the episode.
            episode_id: Episode identifier.

        Returns:
            Questions targeting single events or attributes.
        """
        questions: list[Question] = []
        all_locations = list({event.location for event in events if event.location})
        for event in events:
            if not event.location:
                continue
            location_q = self._create_location_recall_question(
                event, episode_id, all_locations
            )
            if location_q:
                questions.append(location_q)
            subject_q = self._create_subject_recall_question(event, episode_id, events)
            if subject_q:
                questions.append(subject_q)
            event_q = self._create_event_recall_question(event, episode_id, events)
            if event_q:
                questions.append(event_q)
        return questions

    def _create_location_recall_question(
        self, event: Event, episode_id: str, all_locations: list[str]
    ) -> Question | None:
        """Create a location recall question for a given event.

        Args:
            event: Event to question.
            episode_id: Episode identifier.
            all_locations: Candidate locations for distractors.

        Returns:
            Location recall question or None if insufficient options.
        """
        if not event.involved_subjects:
            return None
        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"
        subjects_str = " and ".join(event.involved_subjects)
        question_text = f"Where did {event.event_description.lower()} when {subjects_str} were present?"
        if not self._has_valid_text(question_text):
            return None
        distractors = [
            loc
            for loc in all_locations
            if loc != event.location and self._is_valid_option(loc)
        ]
        if len(distractors) < 3:
            return None
        options_raw = [event.location] + random.sample(distractors, 3)
        options_raw = self._validate_and_deduplicate_options(options_raw)
        if len(options_raw) < 4:
            return None
        options_raw = options_raw[:4]
        answer_index = random.randint(0, 3)
        options: list[str | None] = [None] * 4
        options[answer_index] = event.location
        distractor_idx = 0
        for i in range(4):
            if options[i] is None:
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
            options=[opt or "" for opt in options],
            answer=answer_letter,
            answer_index=answer_index,
            related_events=[event.event_id],
            episode_span=[episode_id],
            clip_span=[event.episode_clip] if event.episode_clip else [],
        )

    def _create_subject_recall_question(
        self, event: Event, episode_id: str, all_events: list[Event]
    ) -> Question | None:
        """Create a subject recall question for a given event.

        Args:
            event: Event to question.
            episode_id: Episode identifier.
            all_events: All events from the episode for distractors.

        Returns:
            Subject recall question or None if invalid.
        """
        if not event.involved_subjects:
            return None
        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"
        question_text = f"Who was present at {event.location} when {event.event_description.lower()}?"
        if not self._has_valid_text(question_text):
            return None
        same_location_events = [
            e
            for e in all_events
            if e.location == event.location
            and e.event_id != event.event_id
            and e.involved_subjects
        ]
        correct_answer = ", ".join(sorted(event.involved_subjects))
        distractors: list[str] = []
        for candidate in same_location_events[:3]:
            distractor = ", ".join(sorted(candidate.involved_subjects))
            if (
                distractor != correct_answer
                and self._is_valid_option(distractor)
                and distractor not in distractors
            ):
                distractors.append(distractor)
        if len(distractors) < 3 and len(event.involved_subjects) > 1:
            variation = ", ".join(event.involved_subjects[:-1])
            if self._is_valid_option(variation):
                distractors.append(variation)
        if len(distractors) < 3:
            return None
        options_raw = [correct_answer] + distractors[:3]
        options_raw = self._validate_and_deduplicate_options(options_raw)
        if len(options_raw) < 4:
            return None
        options_raw = options_raw[:4]
        answer_index = random.randint(0, 3)
        options: list[str | None] = [None] * 4
        options[answer_index] = correct_answer
        distractor_idx = 0
        for i in range(4):
            if options[i] is None:
                options[i] = options_raw[1 + distractor_idx]
                distractor_idx += 1
        answer_letter = chr(65 + answer_index)
        return Question(
            question_id=question_id,
            question_type="single target recall",
            subcategory="subject recall (location + event)",
            cues="L+E",
            target="S",
            question=question_text,
            options=[opt or "" for opt in options],
            answer=answer_letter,
            answer_index=answer_index,
            related_events=[event.event_id],
            episode_span=[episode_id],
            clip_span=[event.episode_clip] if event.episode_clip else [],
        )

    def _create_event_recall_question(
        self, event: Event, episode_id: str, all_events: list[Event]
    ) -> Question | None:
        """Create an event recall question for a given event.

        Args:
            event: Event to question.
            episode_id: Episode identifier.
            all_events: All events from the episode for distractors.

        Returns:
            Event recall question or None if invalid.
        """
        if not event.involved_subjects:
            return None
        self.question_counter += 1
        question_id = f"Q_{episode_id}_{self.question_counter:03d}"
        subjects_str = " and ".join(event.involved_subjects)
        question_text = (
            f"What happened at {event.location} when {subjects_str} were together?"
        )
        if not self._has_valid_text(question_text):
            return None
        same_location_events = [
            e
            for e in all_events
            if e.location == event.location
            and e.event_id != event.event_id
            and self._is_valid_option(e.event_description)
        ]
        distractors = [e.event_description for e in same_location_events[:3]]
        if len(distractors) < 3:
            return None
        options_raw = [event.event_description] + distractors
        options_raw = self._validate_and_deduplicate_options(options_raw)
        if len(options_raw) < 4:
            return None
        options_raw = options_raw[:4]
        answer_index = random.randint(0, 3)
        options: list[str | None] = [None] * 4
        options[answer_index] = event.event_description
        distractor_idx = 0
        for i in range(4):
            if options[i] is None:
                options[i] = options_raw[1 + distractor_idx]
                distractor_idx += 1
        answer_letter = chr(65 + answer_index)
        return Question(
            question_id=question_id,
            question_type="single target recall",
            subcategory="event recall (location + subject)",
            cues="L+S",
            target="E",
            question=question_text,
            options=[opt or "" for opt in options],
            answer=answer_letter,
            answer_index=answer_index,
            related_events=[event.event_id],
            episode_span=[episode_id],
            clip_span=[event.episode_clip] if event.episode_clip else [],
        )

    def _generate_boolean_questions(
        self, events: list[Event], episode_id: str
    ) -> list[Question]:
        """Build yes/no verification questions for locations and subjects.

        Args:
            events: Events from the current episode.
            episode_id: Episode identifier.

        Returns:
            List of boolean `Question` instances.
        """
        questions: list[Question] = []
        all_locations = list({event.location for event in events if event.location})
        all_subjects = list({s for event in events for s in event.involved_subjects})
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
                        explanation=(
                            f"{event.event_description} occurred at {event.location}, not {wrong_loc}."
                        ),
                        related_events=[event.event_id],
                        episode_span=[episode_id],
                        clip_span=[event.episode_clip] if event.episode_clip else [],
                    )
                )
            absent_subjects = [
                subject
                for subject in all_subjects
                if subject not in event.involved_subjects
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
        events_by_episode: dict[str, list[Event]],
        current_episode_id: str,
        max_episode_span: int = 3,
        temporal_questions_per_span: int = 2,
    ) -> list[Question]:
        """Create temporal ordering questions across episode spans.

        Args:
            events_by_episode: Mapping from episode id to its events.
            current_episode_id: Episode id being processed.
            max_episode_span: Maximum span of episodes to include.
            temporal_questions_per_span: Number of questions per span size.

        Returns:
            List of chronological ordering questions.
        """
        questions: list[Question] = []
        current_events = events_by_episode.get(current_episode_id, [])
        if len(current_events) < 2:
            return questions
        episode_ids = sorted(events_by_episode.keys())
        current_idx = episode_ids.index(current_episode_id)
        for span in range(1, max_episode_span + 1):
            start_idx = max(0, current_idx - span + 1)
            end_idx = min(len(episode_ids), current_idx + 1)
            relevant_episode_ids = episode_ids[start_idx:end_idx]
            if len(relevant_episode_ids) < span:
                continue
            span_events: list[Event] = []
            for ep_id in relevant_episode_ids:
                span_events.extend(events_by_episode.get(ep_id, []))
            sorted_events = sorted(span_events, key=lambda event: event.event_id)
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
                if not self._has_valid_text(question_text):
                    continue
                distractors = []
                for candidate in sorted_events:
                    if candidate.event_id not in {
                        event_a.event_id,
                        event_b.event_id,
                    } and self._is_valid_option(candidate.event_description):
                        distractors.append(candidate.event_description)
                    if len(distractors) >= 3:
                        break
                if len(distractors) < 3:
                    continue
                options_raw = [event_a.event_description] + distractors[:3]
                options_raw = self._validate_and_deduplicate_options(options_raw)
                if len(options_raw) < 4:
                    continue
                options_raw = options_raw[:4]
                answer_index = random.randint(0, 3)
                options: list[str | None] = [None] * 4
                options[answer_index] = event_a.event_description
                distractor_idx = 0
                for i in range(4):
                    if options[i] is None:
                        options[i] = options_raw[1 + distractor_idx]
                        distractor_idx += 1
                answer_letter = chr(65 + answer_index)
                episode_span = sorted(
                    {event_a.event_id.split("_")[0], event_b.event_id.split("_")[0]}
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
                        options=[opt or "" for opt in options],
                        answer=answer_letter,
                        answer_index=answer_index,
                        related_events=[event_a.event_id, event_b.event_id],
                        episode_span=episode_span,
                        clip_span=clip_span,
                    )
                )
        return questions

    def _generate_latest_event_retrieval(
        self, events: list[Event], episode_id: str
    ) -> list[Question]:
        """Create questions asking for the latest event in a location or episode.

        Args:
            events: Events from the current episode.
            episode_id: Episode identifier.

        Returns:
            Questions targeting most recent events.
        """
        questions: list[Question] = []
        if not events:
            return questions
        events_by_location: dict[str, list[Event]] = defaultdict(list)
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
            if not self._has_valid_text(question_text):
                continue
            distractors = [
                event.event_description
                for event in loc_events[:-1]
                if self._is_valid_option(event.event_description)
            ][-3:]
            if len(distractors) < 3:
                continue
            options_raw = [latest_event.event_description] + distractors
            options_raw = self._validate_and_deduplicate_options(options_raw)
            if len(options_raw) < 4:
                continue
            options_raw = options_raw[:4]
            answer_index = random.randint(0, 3)
            options: list[str | None] = [None] * 4
            options[answer_index] = latest_event.event_description
            distractor_idx = 0
            for i in range(4):
                if options[i] is None:
                    options[i] = options_raw[1 + distractor_idx]
                    distractor_idx += 1
            answer_letter = chr(65 + answer_index)
            questions.append(
                Question(
                    question_id=question_id,
                    question_type="temporal: latest event retrieval",
                    subcategory="location-based latest event",
                    question=question_text,
                    options=[opt or "" for opt in options],
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
            if not self._has_valid_text(question_text):
                return questions
            distractors = [
                event.event_description
                for event in events[:-1]
                if self._is_valid_option(event.event_description)
            ][-3:]
            if len(distractors) < 3:
                return questions
            options_raw = [latest_event.event_description] + distractors
            options_raw = self._validate_and_deduplicate_options(options_raw)
            if len(options_raw) < 4:
                return questions
            options_raw = options_raw[:4]
            answer_index = random.randint(0, 3)
            options: list[str | None] = [None] * 4
            options[answer_index] = latest_event.event_description
            distractor_idx = 0
            for i in range(4):
                if options[i] is None:
                    options[i] = options_raw[1 + distractor_idx]
                    distractor_idx += 1
            answer_letter = chr(65 + answer_index)
            questions.append(
                Question(
                    question_id=question_id,
                    question_type="temporal: latest event retrieval",
                    subcategory="episode-wide latest event",
                    question=question_text,
                    options=[opt or "" for opt in options],
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
        events: list[Event],
        episode_id: str,
        num_questions: int = 2,
        num_events: int = 4,
    ) -> list[Question]:
        """Create temporal ordering questions using a single cue (location).

        Args:
            events: Events from the current episode.
            episode_id: Episode identifier.
            num_questions: Maximum number of questions to generate.
            num_events: Number of events to order within each question.

        Returns:
            Temporal retrieval questions constrained to one cue.
        """
        questions: list[Question] = []
        if not events:
            return questions
        events_data: list[dict[str, object]] = []
        for event in events:
            if not self._is_valid_option(event.event_description):
                continue
            if not self._is_valid_option(event.location):
                continue
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
        locations: dict[str, list[dict[str, object]]] = defaultdict(list)
        for event_info in events_data:
            if event_info["location"]:
                locations[event_info["location"]].append(event_info)
        valid_locations = [
            loc
            for loc, evs in locations.items()
            if len({e["scene_id"] for e in evs}) > 1 and len(evs) >= num_events
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
            if not self._is_valid_option(location):
                continue
            loc_events = locations[location]
            scenes = list({event_info["scene_id"] for event_info in loc_events})
            random.shuffle(scenes)
            chosen_events: list[dict[str, object]] = []
            used_scenes = set()
            for event_info in random.sample(
                loc_events, min(len(loc_events), num_events * 2)
            ):
                if event_info["scene_id"] not in used_scenes:
                    chosen_events.append(event_info)
                    used_scenes.add(event_info["scene_id"])
                if len(chosen_events) == num_events:
                    break
            if len(chosen_events) < num_events:
                remaining = [ev for ev in loc_events if ev not in chosen_events]
                chosen_events += random.sample(
                    remaining, min(len(remaining), num_events - len(chosen_events))
                )
            if len(chosen_events) < 2:
                continue
            chosen_events = chosen_events[:num_events]
            labeled = {
                f"event_{index + 1}": chosen_events[index]["event_description"]
                for index in range(len(chosen_events))
            }
            labeled_ids = {
                f"event_{index + 1}": chosen_events[index]["event_id"]
                for index in range(len(chosen_events))
            }
            labeled_start = {
                f"event_{index + 1}": chosen_events[index]["start_seconds"]
                for index in range(len(chosen_events))
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
                options_sequences.append(perm)
                if len(options_sequences) >= 4:
                    break
            answer_index = random.randint(0, 3)
            final_options: list[list[str] | None] = [None] * 4
            final_options[answer_index] = correct_sequence
            opt_idx = 0
            for i in range(4):
                if final_options[i] is None:
                    if opt_idx + 1 < len(options_sequences):
                        final_options[i] = options_sequences[opt_idx + 1]
                        opt_idx += 1
                    else:
                        perm = correct_sequence.copy()
                        random.shuffle(perm)
                        final_options[i] = perm
            letters = ["A", "B", "C", "D"]
            mc_options = [
                f"{letters[i]}. {', '.join(final_options[i] or [])}" for i in range(4)
            ]
            answer_letter = letters[answer_index]
            if not self._has_valid_text(" ".join(mc_options)):
                continue
            self.question_counter += 1
            question_id = f"Q_{episode_id}_{self.question_counter:03d}"
            questions.append(
                Question(
                    question_id=question_id,
                    question_type="temporal: single cue retrieval",
                    subcategory="event sequence (location-based)",
                    cues="L",
                    target="Temporal",
                    question=(
                        f"Here are a series of {len(chosen_events)} events. "
                        f"Arrange them in the correct temporal order in which they occurred at {location}.\n\n"
                        + "\n".join(
                            [f'"{label}": {desc}' for label, desc in labeled.items()]
                        )
                    ),
                    options=mc_options,
                    answer=answer_letter,
                    answer_index=answer_index,
                    related_events=[
                        str(labeled_ids[f"event_{idx + 1}"])
                        for idx in range(len(chosen_events))
                    ],
                    episode_span=[episode_id],
                    clip_span=list(
                        {
                            str(event_info["scene_id"]).replace(f"{episode_id}_", "")
                            for event_info in chosen_events
                        }
                    ),
                )
            )
        return questions


def load_episode_data(json_path: Path) -> EpisodeData:
    """Load episode data from a JSON file.

    Args:
        json_path: Path to the episode JSON.

    Returns:
        Parsed episode dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contents are invalid JSON.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: object, output_path: Path, indent: int = 2) -> None:
    """Save data to a JSON file with pretty formatting.

    Args:
        data: Serializable object to write.
        output_path: Destination file path.
        indent: Number of spaces to indent the JSON.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)


def calculate_statistics(questions: list[Question]) -> dict[str, object]:
    """Calculate statistics for generated questions.

    Args:
        questions: List of generated questions.

    Returns:
        Dictionary summarizing distributions across question attributes.
    """
    question_type_distribution: defaultdict[str, int] = defaultdict(int)
    subcategory_distribution: defaultdict[str, int] = defaultdict(int)
    cue_distribution: defaultdict[str, int] = defaultdict(int)
    target_distribution: defaultdict[str, int] = defaultdict(int)
    episode_span_distribution: defaultdict[str, int] = defaultdict(int)
    clip_span_distribution: defaultdict[str, int] = defaultdict(int)
    answer_distribution: defaultdict[str, int] = defaultdict(int)

    for question in questions:
        question_type_distribution[question.question_type] += 1
        subcategory_distribution[question.subcategory] += 1

        if question.cues:
            cue_distribution[f"{question.cues} -> {question.target}"] += 1

        if question.target:
            target_distribution[question.target] += 1

        if question.episode_span:
            episode_span_distribution[f"{len(question.episode_span)} episode(s)"] += 1

        if question.clip_span:
            clip_span_distribution[f"{len(question.clip_span)} clip(s)"] += 1

        if question.answer_index >= 0 and question.answer_index < 4:
            answer_distribution[chr(65 + question.answer_index)] += 1

    return {
        "total_questions": len(questions),
        "question_type_distribution": dict(question_type_distribution),
        "subcategory_distribution": dict(subcategory_distribution),
        "cue_distribution": dict(cue_distribution),
        "target_distribution": dict(target_distribution),
        "episode_span_distribution": dict(episode_span_distribution),
        "clip_span_distribution": dict(clip_span_distribution),
        "answer_distribution": dict(answer_distribution),
    }


def generate_qa_for_file(
    input_file: Path,
    output_file: Path,
    generator: QuestionGenerator,
    max_episode_span: int = 3,
    temporal_questions_per_span: int = 2,
) -> bool:
    """Generate QA dataset for a single enhanced episode file.

    Args:
        input_file: Path to episode JSON containing events.
        output_file: Destination path for the QA JSON.
        generator: Question generator instance.
        max_episode_span: Max span of episodes for temporal questions.
        temporal_questions_per_span: Number of temporal questions per span size.

    Returns:
        True if QA generation succeeds; otherwise False.
    """
    try:
        episode_data = load_episode_data(input_file)
        episode_id = episode_data["episode"]

        events_by_episode: dict[str, list[Event]] = {episode_id: []}
        for scene in episode_data.get("scenes", []):
            for event in scene.get("events", []):
                events_by_episode[episode_id].append(
                    Event(**event) if isinstance(event, dict) else event  # type: ignore[arg-type]
                )

        questions = generator.generate_all_questions(
            episode_data,
            events_by_episode,
            episode_id,
            max_episode_span,
            temporal_questions_per_span,
        )
        stats = calculate_statistics(questions)

        qa_dataset = {
            "episode": episode_id,
            "title": episode_data.get("title", ""),
            "qa_dataset": [asdict(question) for question in questions],
            "statistics": stats,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(qa_dataset, output_file)
        print(f"Saved QA: {output_file}")
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error generating QA for {input_file}: {exc}")
        return False


def generate_qa_for_directory(
    input_dir: Path,
    output_dir: Path,
    max_files: int | None,
    max_episode_span: int,
    temporal_questions_per_span: int,
) -> dict[str, int]:
    """Generate QA datasets for all enhanced episode files in a directory.

    Args:
        input_dir: Directory of episode JSON files with events.
        output_dir: Destination directory for QA JSON files.
        max_files: Optional limit on number of files to process.
        max_episode_span: Max span of episodes for temporal questions.
        temporal_questions_per_span: Number of temporal questions per span size.

    Returns:
        Summary statistics of processing results.
    """
    generator = QuestionGenerator()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if max_files:
        json_files = json_files[:max_files]

    episodes_data: dict[str, EpisodeData] = {}
    events_by_episode: dict[str, list[Event]] = {}
    for json_file in json_files:
        episode_data = load_episode_data(json_file)
        episode_id = episode_data["episode"]
        episodes_data[episode_id] = episode_data
        events: list[Event] = []
        for scene in episode_data.get("scenes", []):
            for event in scene.get("events", []):
                events.append(Event(**event) if isinstance(event, dict) else event)  # type: ignore[arg-type]
        events_by_episode[episode_id] = events

    stats = {"total_files": len(json_files), "processed": 0, "failed": 0}

    for json_file in json_files:
        episode_data = load_episode_data(json_file)
        episode_id = episode_data["episode"]
        questions = generator.generate_all_questions(
            episode_data,
            events_by_episode,
            episode_id,
            max_episode_span,
            temporal_questions_per_span,
        )
        qa_dataset = {
            "episode": episode_id,
            "title": episode_data.get("title", ""),
            "qa_dataset": [asdict(question) for question in questions],
            "statistics": calculate_statistics(questions),
        }
        output_path = output_dir / f"{episode_id}_qa.json"
        try:
            save_json(qa_dataset, output_path)
            stats["processed"] += 1
            print(f"Saved QA: {output_path}")
        except Exception as exc:  # pragma: no cover - defensive
            stats["failed"] += 1
            print(f"Error saving QA for {json_file}: {exc}")

    return stats


def add_events_to_episode(
    episode_data: EpisodeData,
    llm_extractor: LLMEventExtractor,
    max_events_per_scene: int = 10,
    verbose: bool = True,
) -> EpisodeData:
    """Add events to each scene in an episode.

    Args:
        episode_data: Episode with scenes to enrich.
        llm_extractor: Event extractor instance.
        max_events_per_scene: Max events to extract per scene.
        verbose: Whether to print progress.

    Returns:
        Episode data with events added to scenes.
    """
    episode_id = episode_data.get("episode", "unknown")
    scenes = episode_data.get("scenes", [])

    if verbose:
        print(f"\nProcessing Episode {episode_id}")
        print(f"  Title: {episode_data.get('title', 'N/A')}")
        print(f"  Scenes: {len(scenes)}")

    total_events = 0
    scene_iterator = tqdm(
        scenes, desc="  Extracting events", disable=not verbose, leave=False
    )

    for scene_idx, scene in enumerate(scene_iterator, start=1):
        events = llm_extractor.extract_events(
            scene, episode_id, scene_idx, max_events=max_events_per_scene
        )
        scene["events"] = [asdict(event) for event in events]
        total_events += len(events)
        if verbose:
            scene_iterator.set_postfix({"total_events": total_events})

    if verbose and scenes:
        avg_events = total_events / len(scenes)
        print(f"  Extracted {total_events} events total")
        print(f"  Average {avg_events:.1f} events per scene")

    return episode_data


def process_single_file(
    input_path: Path,
    output_path: Path,
    llm_extractor: LLMEventExtractor,
    max_events_per_scene: int = 10,
    overwrite: bool = False,
) -> bool:
    """Process a single JSON file.

    Args:
        input_path: Path to the source episode JSON.
        output_path: Destination for the enriched episode JSON.
        llm_extractor: Event extractor to use.
        max_events_per_scene: Max events to extract per scene.
        overwrite: Whether to replace an existing output file.

    Returns:
        True if processing succeeds; otherwise False.
    """
    if output_path.exists() and not overwrite:
        print(f"Skipped: Output file already exists: {output_path}")
        print("  Use --overwrite to replace it")
        return False

    try:
        episode_data = load_episode_data(input_path)
        enhanced_data = add_events_to_episode(
            episode_data,
            llm_extractor,
            max_events_per_scene=max_events_per_scene,
            verbose=True,
        )
        save_json(enhanced_data, output_path)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error processing {input_path}: {exc}\n")
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    llm_extractor: LLMEventExtractor,
    max_events_per_scene: int = 10,
    max_files: int | None = None,
    overwrite: bool = False,
    resume: bool = True,
) -> dict[str, int]:
    """Process all JSON files in a directory.

    Args:
        input_dir: Directory containing episode JSON files.
        output_dir: Destination for enriched JSON outputs.
        llm_extractor: Event extractor instance.
        max_events_per_scene: Max events to extract per scene.
        max_files: Optional limit on files to process.
        overwrite: Whether to replace existing outputs.
        resume: Skip files already processed when True.

    Returns:
        Processing statistics across all files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(Path(input_dir).glob("*.json"))
    if max_files:
        json_files = json_files[:max_files]

    print(f"Processing {len(json_files)} files from {input_dir}")
    print(f"Output directory: {output_dir}")

    stats = {"total_files": len(json_files), "processed": 0, "skipped": 0, "failed": 0}

    if resume:
        processed_files = {
            file.stem.replace("_with_events", "")
            for file in output_dir.glob("*_with_events.json")
        }
        if processed_files:
            print(f"Resume mode: Found {len(processed_files)} already processed files")
            json_files = [f for f in json_files if f.stem not in processed_files]
            stats["skipped"] = len(processed_files)
            print(f"Will process {len(json_files)} remaining files\n")

    for idx, json_file in enumerate(json_files, start=1):
        output_filename = f"{json_file.stem}_with_events.json"
        output_path = output_dir / output_filename

        success = process_single_file(
            json_file,
            output_path,
            llm_extractor,
            max_events_per_scene=max_events_per_scene,
            overwrite=overwrite,
        )

        if success:
            stats["processed"] += 1
        else:
            stats["failed"] += 1

    print("PROCESSING COMPLETE:")
    print(f"Total files:     {stats['total_files']}")
    print(f"Processed:       {stats['processed']}")
    print(f"Skipped:         {stats['skipped']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Output dir:      {output_dir}")

    return stats


def process_pipeline(
    input_file: Path | None = None,
    output_file: Path | None = None,
) -> dict[str, dict[str, int] | bool]:
    """Run event extraction then QA generation using the data/* defaults (no CLI).

    Args:
        input_file: Optional path to process a single episode JSON.
        output_file: Output path matching `input_file` when processing one file.

    Returns:
        Mapping with event processing stats and QA generation results.

    Raises:
        ValueError: If `output_file` is missing when `input_file` is provided.
    """
    config = ProcessConfig()
    llm_extractor = LLMEventExtractor(
        config.model_path,
        device=config.device,
        backend=config.llm_backend,
        openai_model=config.openai_model,
        temperature=config.temperature,
    )

    results: dict[str, dict[str, int] | bool] = {}

    if input_file:
        if not output_file:
            raise ValueError("output_file is required when processing a single file.")
        event_result = process_single_file(
            input_file,
            output_file,
            llm_extractor,
            max_events_per_scene=config.max_events,
            overwrite=config.overwrite,
        )
        results["events"] = event_result

        qa_output = config.qa_output_dir / (
            f"{Path(output_file).stem.replace('_with_events', '')}_qa.json"
        )
        qa_success = generate_qa_for_file(
            output_file,
            qa_output,
            QuestionGenerator(),
            max_episode_span=config.max_episode_span,
            temporal_questions_per_span=config.temporal_questions_per_span,
        )
        results["qa"] = {
            "processed": int(bool(qa_success)),
            "failed": int(not qa_success),
        }
        return results

    event_stats = process_directory(
        config.input_dir,
        config.output_dir,
        llm_extractor,
        max_events_per_scene=config.max_events,
        max_files=config.max_files,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    results["events"] = event_stats

    qa_stats = generate_qa_for_directory(
        config.output_dir,
        config.qa_output_dir,
        max_files=config.max_files,
        max_episode_span=config.max_episode_span,
        temporal_questions_per_span=config.temporal_questions_per_span,
    )
    results["qa"] = qa_stats
    return results
