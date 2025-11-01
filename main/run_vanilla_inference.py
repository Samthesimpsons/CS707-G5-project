import json
import torch
import gc
import huggingface_hub

from pathlib import Path
from models.inference import EpisodicInference
from models.model import QwenVL, VideoLlava, VideoLLama2, VideoLLama3

HF_CACHE_DIR = Path("/common/home/projectgrps/CS707/CS707G3/.cache/huggingface/hub")

QA_PAIRS_DIR = r"./data/episodic_qa/"
TUPLES_DIR = r"./data/episodic_tuples/"
VIDEO_DIR = r"./data/video/"

MODEL_PATH = {
    "qwen-vl-2": "/common/public/Qwen2.5-VL/Qwen2-VL-7B-Instruct",
    "qwen-vl-25": "/common/public/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct",
    "video-llava": "LanguageBind/Video-LLaVA-7B-hf",
    "video-llama-2": "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base",
    "video-llama-3": "DAMO-NLP-SG/VideoLLaMA3-7B"
}

MODEL_CLASSES = {
    "qwen-vl-2": QwenVL,
    "qwen-vl-25": QwenVL,
    "video-llava": VideoLlava,
    "video-llama-2": VideoLLama2,
    "video-llama-3": VideoLLama3
}


if __name__ == "__main__":
    for model_name, model_path in MODEL_PATH.items():
        print(f"\n{'='*80}")
        print(f"Running inference for model: {model_name}")
        print(f"Model path: {model_path}")
        print(f"{'='*80}\n")
        
        gc.collect()
        torch.cuda.empty_cache()
        print("Torch CUDA Cache Emptied")
    
        # MODEL_PATH = "LanguageBind/Video-LLaVA-7B-hf"
        # MODEL_PATH = "/common/public/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct"
        # model = QwenVL(model_path=MODEL_PATH)
        # model = VideoLlava(model_path=MODEL_PATH)

        ### Initialize the Models
        print(f"Inference: Initializing Model {model_path}")
        model_class = MODEL_CLASSES.get(model_name)
        if model_class is None:
            available_models = [k for k in MODEL_CLASSES.keys()]
            raise ValueError(f"No model class found for {model_name} - Available models {available_models}")
        model = model_class(model_path=model_path)
        print(f"Inference: Model Loaded from {model_path}")
            
        ### Begin running the Inference
        print(f"Inference: Begin Run")
        inference = EpisodicInference(
            model = model,
            episodic_tuples_dir = TUPLES_DIR,
            qa_pairs_dir = QA_PAIRS_DIR,
            video_dir = VIDEO_DIR,
            output_path = r"./results/",
            with_subs = False,
            all_subs = False
        )
        inference.generate_batch_run()