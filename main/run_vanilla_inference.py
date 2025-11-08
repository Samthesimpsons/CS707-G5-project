import json
import torch
import gc
import huggingface_hub
import random
import numpy as np

from pathlib import Path
from models.inference import EpisodicInference
from models.model import QwenVL, VideoLlava, VideoLLama3
from datetime import datetime


HF_CACHE_DIR = Path("/common/home/projectgrps/CS707/CS707G3/.cache/huggingface/hub")
QA_PAIRS_DIR = r"./data/episodic_qa/"
TUPLES_DIR = r"./data/episodic_tuples/"
VIDEO_DIR = r"./data/video/"
OUTPUT_SAVE_DIR = r"./results_vanilla/"


MODEL_PATH = {
    # "qwen-vl-2": "/common/public/Qwen2-VL/Qwen2-VL-7B-Instruct",
    # "qwen-vl-25": "/common/public/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct",
    "video-llama-3": "DAMO-NLP-SG/VideoLLaMA3-7B",
    "video-llava": "LanguageBind/Video-LLaVA-7B-hf"
}


MODEL_CLASSES = {
    "qwen-vl-2": QwenVL,
    "qwen-vl-25": QwenVL,
    "video-llava": VideoLlava,
    "video-llama-3": VideoLLama3
}

"""
srun --pty --qos=cs707qos --partition=project --gres=gpu:1 bash
"""
# def set_experiment_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuca.manual_all_seed(seed)
#     print(f"Seed set to {seed}")


if __name__ == "__main__":
    # set_experiment_seed(seed)
    for model_name, model_path in MODEL_PATH.items():
        run_datetime = datetime.now().strftime("%Y%m%d_%H%M")
        # for run_id in [1, 2, 3]:
        for run_id in [1]:
            print(f"\n{'='*80}")
            print(f"Running Inference for model: {model_name}  -- {run_datetime} -- Run ID: {run_id}")
            print(f"Model path: {model_path}")
            print(f"{'='*80}\n")
            gc.collect()
            torch.cuda.empty_cache()
            print("Torch CUDA Cache Emptied")

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
                output_path = OUTPUT_SAVE_DIR,
                with_subs = False,
                all_subs = False,
                with_context = False,
                run_id = str(run_id),
                run_datetime = run_datetime
            )
            inference.generate_batch_run()

                