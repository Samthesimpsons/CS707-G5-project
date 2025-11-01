import av
import json
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

from datetime import datetime
from dataclasses import dataclass

print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU only")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


@dataclass
class EpisodicQA:
    pass


@dataclass
class EpisodicTuples:
    pass


class EpisodicInference:
    def __init__ (
        self, 
        model, 
        episodic_tuples_dir, # input path to the episodic tuples
        qa_pairs_dir, # input path to the QA pairs
        video_dir, # input path to the videos
        output_path: str, # output path to save the response
        with_subs: bool, # to include subs or not
        all_subs: bool # to extract subs for full episode (True) or event-specific
    ):
        self.episodic_tuples_dir = episodic_tuples_dir
        self.qa_pairs_dir = qa_pairs_dir
        self.qa_pairs = self.load_question_answer_pairs(qa_pairs_dir)
        self.video_dir = video_dir
        self.model = model
        self.output_path = output_path
        self.with_subs = with_subs
        self.all_subs = all_subs
        self.responses = []

    
    ### ==========================================
    ### QUESTION ANSWER - JSON HANDLING AND SAVING
    ### ==========================================
    def load_question_answer_pairs(self, qa_pairs_dir) -> List[Dict]:
        path = Path(qa_pairs_dir)
        entries = []
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
        elif path.is_dir():
            for f in sorted(path.glob("*.json")):
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    entries += data if isinstance(data, list) else [data]
        else:
            raise FileNotFoundError(f"QA file/dir not found: {path}")
        return entries

    ### ==========================================
    ### PROMPT BUILDER
    ### ==========================================
    def extract_video_clips_paths(self, qa_json: dict) -> list:
        """
            Output is a list of video clip paths for the given QA JSON
        """
        video_paths = []
        episodes = qa_json.get("episode_span", [])
        for ep in episodes:
            season_id = int(ep[:2])
            clip_id = int(ep[2:])
            clip_path = Path(self.video_dir) / f"season_{season_id}" / f"episode_{clip_id}.mp4"
            video_paths.append(str(clip_path))
        return video_paths


    def extract_subtitles(self, qa_json: dict, episode_id: int) -> list:
        """
            Extract all corresponding subtitles to be used as part of prompt for QA to add context
        """
        # episode_id = qa_json.get("episode")
        # if not episode_id:
        #     raise ValueError("QA JSON missing 'episode' field.")
        
        # Locate episodic tuple file
        episodic_tuples_path = Path(self.episodic_tuples_dir) / f"friends_{episode_id}_with_events.json"
        if not episodic_tuples_path.exists():
            raise FileNotFoundError(f"Episodic tuple not found for episode {episode_id}: {episodic_tuples_path}")

        ### Load the episode's tuple
        with open(episodic_tuples_path, "r", encoding="utf-8") as f:
            episode_tuple = json.load(f)
        assert episode_tuple.get("episode") == episode_id, (
            f"Episode mismatch: expected {episode_id}, got {episode_tuple.get('episode')}"
        )
        ### Extract subtitles for all relevant events
        subtitles_output = []
        related_events = qa_json.get("related_events", [])
        collected_dialogues = []

        if self.all_subs:
            # Extract *all* subtitles from all scenes
            for scene in episode_tuple.get("scenes", []):
                for event_dialogue in scene.get("dialogue", []):
                    speaker = event_dialogue.get("speaker", "").strip()
                    text = event_dialogue.get("text", "").strip()
                    if speaker and text:
                        collected_dialogues.append(f"{speaker}: {text}")
        else:
            # Extract only subtitles tied to related events
            related_events = qa_json.get("related_events", [])
            for scene in episode_tuple.get("scenes", []):
                # Check if event_id in related_events are present in the episode tuple
                scene_event_ids = [e["event_id"] for e in scene.get("events", [])]
                if any(event in scene_event_ids for event in related_events):
                    
                    for event_dialogue in scene.get("dialogue", []):
                        speaker = event_dialogue.get("speaker", "").strip()
                        text = event_dialogue.get("text", "").strip()
                        if speaker and text:
                            collected_dialogues.append(f"{speaker}: {text}")
        
        # # Loop through all the scenes        
        # for scene in episode_tuple.get("scenes", []):
        #     scene_event_ids = [e["event_id"] for e in scene.get("events", [])]
            
        #     if any(event in scene_event_ids for event in related_events):
        #         scene_description = scene.get("scene_description", "")

        #         # extract the full dialogue for this scene
        #         for event_dialogue in scene.get("dialogue", []):
        #             speaker = event_dialogue.get("speaker", "").strip()
        #             text = event_dialogue.get("text", "").strip()
        #             if speaker and text:
        #                 collected_dialogues.append(f"{speaker}: {text}")
                        
        # Combine all subtitles into a list
        subtitles_output.append(collected_dialogues)
        return subtitles_output


    def build_prompt_text(self, qa_json: dict, episode_id: int) -> str:
        """
            Generates a prompt that is fed into the MLLM during inference
        """
        prompt_parts = []
        model_name = self.model.model_path.split("/")[-1]
        instruction = "Respond only with the number of the correct multiple-choice answer option (e.g., 1)."
        
        if self.with_subs:
            subs = self.extract_subtitles(qa_json, episode_id)
            subs_text = str(subs)

        # Generate questions and answers
        question = qa_json.get("question", "")
        question_id = qa_json.get("question_id", "")
        options_text = "\n".join(f"{i + 1}. {place}" for i, place in enumerate(qa_json.get("options", [])))

        if self.with_subs:
            prompt_parts.append(
                f"\n\nVideo Context (Subtitles):\n{subs_text}\n\n"
                f"Question:\n{question}\n\n"
                f"Answer Options:\n{options_text}\n"
                f"{instruction}\n\n"
            )
        else:
            prompt_parts.append(
                f"Question:\n{question}\n\n"
                f"Answer Options:\n{options_text}\n"
                f"{instruction}\n\n"
            )

        # if "qwen" in model_name.lower():
        #     prompt_parts.append("""\no__think""")

        prompt = "\n\n".join(prompt_parts)
        return prompt, options_text


    ### ==========================================
    ### BATCH JOB RUNNER
    ### ==========================================
    def generate_batch_run(self, max_new_tokens: int = 128) -> List[Dict]:
        if self.output_path is None:
            raise RuntimeError("Output Save Path for Results is Not Specified")
        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        print(f"\n{'='*80}")
        print(f"Batch Job Inference: Initializing --- {now_str}")
        print(f"{'='*80}\n")

        # Begin the batch inference
        results = []
        model_name = self.model.model_path.split("/")[-1]
        
        for idx, qa_json in enumerate(tqdm(self.qa_pairs, desc="EpisodicInference Batch Run")):
            episode_id = qa_json.get("episode")
            print(f"Generating Batch Inference - Index: {idx} --- Episode: {episode_id}")
            qa_dataset = qa_json.get("qa_dataset", [])

            for idx, qa_data in enumerate(tqdm(qa_dataset, desc=f"Running QA Dataset - {episode_id}")):
                question_id = qa_data.get("question_id")
                text_prompt, options_text = self.build_prompt_text(qa_data, episode_id)
                video_paths = self.extract_video_clips_paths(qa_data)
                
                print(f"\n============{question_id}============")
                print(f"\n{qa_data}\n")
                response = {
                    "model": model_name,
                    "question": qa_data["question"],
                    "question_id": qa_data["question_id"],
                    "question_type": qa_data["question_type"],
                    "ground_truth_answer": qa_data["answer"],
                    "options_text": options_text,
                    "model_response": "",
                    "execution_time": "",
                    "video_clips": "",
                    "error": "",
                }
                try:
                    # video_paths = [r"./src/data/video_clip/0102_scene_000_central_perk.mp4"]
                    print(f"Videos to run inference on: {video_paths}")
                    generated_result, execution_time = self.model.run_inference(
                        text_prompt = text_prompt,
                        video_paths = video_paths,
                        max_new_tokens = max_new_tokens
                    )
                    response["model_response"] = generated_result
                    response["execution_time"] = execution_time
                    response["video_clips"] = video_paths
                except Exception as e:
                    response["error"] = f"{e}"
                    raise
                
                qa_data["result"] = response
                results.append(qa_data)

            # Save the results
            output_path = Path(self.output_path) / model_name
            output_path.mkdir(exist_ok=True)
            save_path = output_path / f"results_{Path(self.qa_pairs_dir).name}_{now_str}.json"
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"{self.qa_pairs_dir} Saved results to: {save_path}")
            print(f"Ending Batch Inference - Index: {idx} --- Episode: {episode_id}")
            torch.cuda.empty_cache()

        return results