
import json

from pathlib import Path

"""
    Design evaluation metrics
    TODO: Accuracy
    TODO: Precision, Recall , F1
    TODO: Exact Match
    TODO: BLEU, ROUGE
    TODO: Kendall's Tau
"""

### SAMPLE RESULT OUTPUT
"""
{
        "question_id": "Q_0102_001",
        "question_type": "single target recall",
        "subcategory": "location recall (event + subject)",
        "cues": "E+S",
        "target": "L",
        "question": "Where did Monica explain that kissing is important for women when Rachel and Phoebe were present?",
        "options": [
            "Central Perk",
            "Monica and Rachel's apartment",
            "Museum of Prehistoric History",
            "Carol's OB/GYN"
        ],
        "answer": "Central Perk",
        "answer_index": 0,
        "related_events": [
            "0102_scene_000_event_001"
        ],
        "episode_span": [
            "0102"
        ],
        "result": {
            "model": "Qwen2.5-VL-7B-Instruct",
            "question": "Where did Monica explain that kissing is important for women when Rachel and Phoebe were present?",
            "question_id": "Q_0102_001",
            "question_type": "single target recall",
            "ground_truth_answer": "Central Perk",
            "options_text": "1. Central Perk\n2. Monica and Rachel's apartment\n3. Museum of Prehistoric History\n4. Carol's OB/GYN",
            "model_response": [
                "1"
            ],
            "execution_time": "51.8550",
            "video_clips": [
                "data/video/season_1/episode_2.mp4"
            ],
            "error": ""
        }
    }
"""