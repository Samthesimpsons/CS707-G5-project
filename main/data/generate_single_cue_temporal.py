"""
Generate structured multiple-choice "temporal ordering" questions
from Friends episode JSON data.

Each question:
- Focuses on a single location (e.g., Monica and Rachel's apartment)
- Includes 4 labeled events (event_1–event_4)
- Provides 4 multiple-choice orderings (A–D)
- Identifies the correct chronological order and answer index

Output format:
[
  {
    "question_id": "Q_<episode_number>_<i>",
    "question_type": "temporal ordering",
    "subcategory": "temporal ordering (event + location)",
    "question": "Here are a series of 4 events...",
    "options": [...],
    "answer": "B.",
    "answer_index": 1,
    "related_events": [...],
    "episode_span": ["0102"]
  },
  ...
]
"""

import json
import random
import argparse
from typing import List, Dict


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def hhmmss_to_seconds(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def event_start_seconds(timestamp: str) -> int:
    start = timestamp.split("-")[0].strip()
    return hhmmss_to_seconds(start)


def load_episode_data(filepath: str) -> Dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_events(data: dict) -> List[Dict]:
    """Flatten all scenes and include scene context for cross-scene sampling."""
    events = []
    for scene in data.get("scenes", []):
        location = scene.get("location", "")
        scene_id = scene.get("context_label", "")
        for ev in scene.get("events", []):
            ev_copy = ev.copy()
            ev_copy["start_seconds"] = event_start_seconds(ev_copy["timestamp"])
            ev_copy["location"] = location
            ev_copy["scene_id"] = scene_id
            events.append(ev_copy)
    return events


def generate_event_order_question(
    events: List[Dict],
    episode_number: str,
    q_index: int,
    num_events: int = 4
) -> Dict:
    """Generate one temporal-ordering question sampling across scenes but within the same location."""

    # Pick a random location with events across multiple scenes
    locations = {}
    for ev in events:
        locations.setdefault(ev["location"], []).append(ev)

    # Choose a location that appears in at least 2 different scenes
    valid_locations = [
        loc for loc, evs in locations.items()
        if len(set(e["scene_id"] for e in evs)) > 1 and len(evs) >= num_events
    ]
    if not valid_locations:
        raise ValueError("No multi-scene locations available for sampling.")

    location = random.choice(valid_locations)
    loc_events = locations[location]

    # Ensure we sample from different scenes if possible
    scenes = list({e["scene_id"] for e in loc_events})
    random.shuffle(scenes)
    chosen_events = []
    used_scenes = set()

    for e in random.sample(loc_events, len(loc_events)):
        if e["scene_id"] not in used_scenes or len(chosen_events) < num_events // 2:
            chosen_events.append(e)
            used_scenes.add(e["scene_id"])
        if len(chosen_events) == num_events:
            break

    # Fallback if not enough scenes to diversify
    if len(chosen_events) < num_events:
        remaining = [e for e in loc_events if e not in chosen_events]
        chosen_events += random.sample(remaining, num_events - len(chosen_events))

    # Prepare question
    selected_sorted = sorted(chosen_events, key=lambda e: e["start_seconds"])
    labeled = {f"event_{i+1}": chosen_events[i]["event_description"] for i in range(num_events)}
    labeled_ids = {f"event_{i+1}": chosen_events[i]["event_id"] for i in range(num_events)}
    labeled_start = {f"event_{i+1}": chosen_events[i]["start_seconds"] for i in range(num_events)}

    correct_sequence = [label for label, _ in sorted(labeled_start.items(), key=lambda kv: kv[1])]
    all_labels = list(labeled.keys())
    correct = correct_sequence
    options = [correct.copy()]
    seen = {tuple(correct)}

    while len(options) < 4:
        perm = random.sample(all_labels, len(all_labels))
        if tuple(perm) not in seen:
            options.append(perm)
            seen.add(tuple(perm))

    random.shuffle(options)
    correct_index = options.index(correct)
    letters = ["A", "B", "C", "D"]
    mc_options = [f"{letters[i]}. {', '.join(options[i])}" for i in range(4)]

    question_text = (
        f'Here are a series of {num_events} events. Arrange them in the correct temporal order '
        f'in which they occurred at **{location}**.\n\n' +
        "\n".join([f'"{label}": {desc}' for label, desc in labeled.items()])
    )

    return {
        "question_id": f"Q_{episode_number}_{q_index}",
        "question_type": "temporal ordering",
        "subcategory": "temporal ordering (event + location)",
        "location": location,
        "question": question_text,
        "options": mc_options,
        "answer": f"{letters[correct_index]}.",
        "answer_index": correct_index,
        "related_events": [labeled_ids[f"event_{i+1}"] for i in range(num_events)],
        "episode_span": [episode_number],
    }


def generate_questions(filepath: str, n: int = 3, events_per_question: int = 4) -> List[Dict]:
    data = load_episode_data(filepath)
    episode_number = data.get("episode", "0000")
    events = extract_events(data)

    questions = []
    for i in range(1, n + 1):
        q = generate_event_order_question(events, episode_number, i, events_per_question)
        questions.append(q)
    return questions

# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    file = "./data/episodic_tuples/friends_0102_with_events.json"
    out = "./data/temporal_qa.json"
    output = generate_questions(file, n=3)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(output)} questions → {out}")
