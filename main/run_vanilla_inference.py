import json
import torch
import gc
import huggingface_hub
import random
import numpy as np
import argparse

from pathlib import Path
from models.inference import EpisodicInference
from models.model import QwenVL, VideoLlava, VideoLLama3
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--qa_dir', type=str, default=r"./data/episodic_qa/")
parser.add_argument('--tuples_dir', type=str, default=r"./data/episodic_tuples/")
parser.add_argument('--video_dir', type=str, default=r"./data/video/")
parser.add_argument('--output_dir', type=str, default=r"./results_vanilla/")
parser.add_argument('--with_context', type=bool, default=False)
args = parser.parse_args()

HF_CACHE_DIR = Path("/common/home/projectgrps/CS707/CS707G3/.cache/huggingface/hub")
# QA_PAIRS_DIR = args.qa_dir
# TUPLES_DIR = args.tuples_dir
# VIDEO_DIR = args.video_dir
# OUTPUT_SAVE_DIR = args.output_dir


MODEL_PATH = {
    # "qwen-vl-2": "/common/public/Qwen2-VL/Qwen2-VL-7B-Instruct",
    # "qwen-vl-25": "/common/public/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct",
    "video-llama-3": "DAMO-NLP-SG/VideoLLaMA3-7B",
    # "video-llava": "LanguageBind/Video-LLaVA-7B-hf"
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
                episodic_tuples_dir = args.tuples_dir,
                qa_pairs_dir = args.qa_dir,
                video_dir = args.video_dir,
                output_path = args.output_dir,
                with_subs = False,
                all_subs = False,
                with_context = args.with_context,
                run_id = str(run_id),
                run_datetime = run_datetime,
                checkpoint = args.checkpoint
            )
            inference.generate_batch_run()

                