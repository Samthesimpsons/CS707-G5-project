import av
import json
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

from model import QwenVL
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
            processor, 
            episodic_tuples_dir, # input path to the episodic tuples
            qa_pairs_dir, # input path to the QA pairs
            device: str,
            num_frames: int,
            output_path: str # output path to save the response
    ):
        self.episodic_tuples_dir = episodic_tuples_dir
        self.qa_pairs_dir = qa_pairs_dir
        self.qa_pairs = self.load_question_answer_pairs(qa_pairs_dir)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)

        self.processor = processor
        self.num_frames = num_frames
        self.output_path = output_path
        self.responses = []

    ### ==========================================
    ### VIDEO CLIP PROCESSING
    ### ==========================================
    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`list[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    

    def sample_clip_frames(self, video_path: str) -> np.ndarray:
        """
        Open a video, sample frames, and return them as np.ndarray.
        Args:
            video_path
        Returns:
            result (np.ndarray): np array of clip frames
        """
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = getattr(stream, "frames", None)
        if not total_frames or total_frames <= 0:
            # fallback: decode all frames and sample
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            total_frames = len(frames)
            if total_frames == 0:
                raise RuntimeError(f"No frames found in {video_path}")
            indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
            clip = np.stack([frames[i] for i in indices])
        else:
            indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
            clip = self.read_video_pyav(container, indices.tolist())
        container.close()
        return clip
    
    ### ==========================================
    ### QUESTION ANSWER - JSON HANDLING AND SAVING
    ### ==========================================
    def load_question_answer_pairs(self) -> List[Dict]:
        path = Path(self.qa_pairs_dir)
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
    

    def save_question_answer_response(self) -> Path:
        save_path = Path(self.output_path)
        ### Write self.responses
        pass


    ### ==========================================
    ### PROMPT BUILDER
    ### ==========================================
    def extract_video_clips_paths(self) -> list:
        """
            Output is a list of video clip paths
        """
        pass


    def extract_subtitles(self, qa_json: dict) -> list:
        """
            Extract all corresponding subtitles to be used as part of prompt for QA to add context
        """
        episode_id = qa_json.get("episode")
        if not episode_id:
            raise ValueError("QA JSON missing 'episode' field.")
        
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
        qa_dataset = qa_json.get("qa_dataset", [])
        subtitles_output = []

        for qa_item in qa_dataset:
            related_events = qa_item.get("related_events", [])
            collected_dialogues = []
            # Loop through all the scenes        
            for scene in episode_tuple.get("scenes", []):
                scene_event_ids = [e["event_id"] for e in scene.get("events", [])]
                if any(event in scene_event_ids for event in related_events):
                    # extract the full dialogue for this scene
                    for event_dialogue in scene.get("dialogue", []):
                        speaker = event_dialogue.get("speaker", "").strip()
                        text = event_dialogue.get("text", "").strip()
                        if speaker and text:
                            collected_dialogues.append(f"{speaker}: {text}")
            subtitles_output.append(collected_dialogues)
        return subtitles_output


    def build_prompt(self, qa_json: dict) -> str:
        """
            Generates a prompt that is fed into the MLLM during inference
        """
        prompt_parts = []
        # clips = qa_json.get("video_clips", []) # ignore
        # multi_clip = len(clips) > 1 # ignore
        qa_dataset = qa_json.get("qa_dataset")
        subs = self.extract_subtitles(qa_json)

        subtitle_map = {item["question_id"]: item["subtitles"] for item in subs}

        for idx, qa_data in enumerate(qa_dataset, start=1):
            # Extract all corresponding subtitles
            subs = self.extract_subtitles(qa_json)
            subs_text = "\n".join(subs) if subs else "[No dialogue context found]"

            # Generate questions and answers
            question = qa_data.get("question", "")
            options_text = "\n".join(f"{i + 1}. {place}" for i, place in enumerate(qa_data.get("options", [])))
            
            prompt_parts.append(
                f"Scene Context (Subtitles):\n{subs_text}\n\n"
                f"Question:\n{question}\n\n"
                f"Answer Options:\n{options_text}\n"
            )

        prompt = "\n\n".join(prompt_parts)
        return prompt
    

    ### ==========================================
    ### SINGLE AND MULTI-CLIP INFERENCE
    ### ==========================================
    def process_and_infer(self, qa_json: dict, max_new_tokens: int) -> str:
        """
            MLLM response to the QA pair
        """
        response = {
            "question": qa_json["question"],
            "ground_truth_answer": qa_json["answer"],
            "predicted_response": None,
            "execution_time": None,
            "error": None,
            "video_clips": None
        }
        try:
            # Combine frames from all clips
            clips_data = []
            clips_ls = self.extract_video_clips_paths(qa_json)

            if len(clips_ls) > 1:
                for v in clips_ls:
                    clip_frames = self.sample_clip_frames(v["path"])
                    clips_data.append(clip_frames)
                combined_clip = np.concatenate(clips_data, axis=0)
            else:
                combined_clip = clips_ls[0]

            if not clips_data:
                raise ValueError("No video clips provided in entry.")

            # Prepare the prompt
            prompt = self.build_prompt(qa_json)

            # Prepare the inputs for Model Processor
            start = time.perf_counter()
            inputs = self.processor(
                text=prompt,
                image=image, ### TO CHECK IF CAN PASS LOCATIONS OR SOMETHING HERE?
                videos=combined_clip,
                padding=True,
                return_tensors = "pt"
            ).to(self.device)

            # Start the inference 
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            end = time.perf_counter()

            result["predicted_answer"] = output
            result["execution_time_sec"] = round(end - start, 4)
        
        return result


    ### ==========================================
    ### BATCH JOB RUNNER
    ### ==========================================
    def generate_batch(self, max_new_tokens: int = 256) -> List[Dict]:
        if self.output_path is None:
            raise RuntimeError("Output Save Path for Results is Not Specified")
        
        print(f"\n{'='*80}")
        print("Batch Job Inference: Initializing")
        print(f"{'='*80}\n")

        # Begin the batch inference
        results = []
        for idx, qa_json in enumerate(tqdm(self.qa_pairs, desc="EpisodicInference Batch Run")):
            generated_result = self.process_and_infer(
                qa_json=qa_json,
                max_new_tokens=max_new_tokens
            )
            generated_result["idx"] = idx
            results.append(generated_result)

        # Save the results
        output_path = Path(self.output_path)
        output_path.mkdir(exist_ok=True)

        save_path = output_path / f"results_{Path(self.qa_pairs_dir).name}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"{self.qa_pairs_dir} Saved results to: {save_path}")
        return results

        

        



    # def generate(self, max_new_tokens: int=2048*2):
    #     generated_answers = []
    #     for idx, question in enumerate(self.qa_pairs):
    #         video_path = self.qa_pairs.get("video_path", "")
    #         container = av.open(video_path)

    #         total_frames = container.streams.video[0].frames
    #         indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    #         clip = self.read_video_pyav(container, indices)

    #         inputs = self.processor(text=question, videos=clip, return_tensors="pt")
    #         generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
    #         output_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    #         generated_answers.append(
    #             {
    #                 "idx": idx,
    #                 "question": question,
    #                 "response": output_text
    #             }
    #         )
    
