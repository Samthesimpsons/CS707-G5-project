import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import gc
from goldfish.inference_utils_batch import load_question_answer_pairs, generate_batch
from pathlib import Path
from goldfish.model import VideoLlava, VideoLLama3, QwenVL, InternVL3

#HF_CACHE_DIR = Path("/common/home/users/d/divyam.2023/.cache/huggingface/hub")

QA_PAIRS_DIR = r"./data/episodic_qa/"
#QA_PAIRS_DIR = r"../main/data/Archive/episodic_qa/"
#TUPLES_DIR = r"../main/data/episodic_tuples/"
VIDEO_DIR = r"./data/video/"

MODEL_PATH = {
    # "qwen-vl-2": "/common/public/Qwen2.5-VL/Qwen2-VL-7B-Instruct",
    #"qwen-vl-25": "/common/public/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct",
    "video-llava": "LanguageBind/Video-LLaVA-7B-hf",
    #"video-llama-3": "DAMO-NLP-SG/VideoLLaMA3-7B",
    #"intern-vl-3": "OpenGVLab/InternVL3-8B"
}

MODEL_CLASSES = {
    #"qwen-vl-2": QwenVL,
    #"qwen-vl-25": QwenVL,
    "video-llava": VideoLlava,
    # "video-llama-2": VideoLLama2,
    #"video-llama-3": VideoLLama3,
    #"intern-vl-3": InternVL3
}


max_new_tokens = 128
neighbours = [1,3,5]
use_openai_embedding = False
qa_pairs = load_question_answer_pairs(QA_PAIRS_DIR)
captions = ['generic', 'specific']
#print(qa_pairs)
for model_name, model_path in MODEL_PATH.items():
    print(f"\n{'='*80}")
    print(f"Running inference for model: {model_name}")
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
        
    for r in range(1,2): 
        for c in captions:
            for n in neighbours:
                generate_batch(model, qa_pairs, QA_PAIRS_DIR, VIDEO_DIR,  output_path=f'./results/{model_name}_final/run_{str(r)}/caption_{c}/nn_{str(n)}',max_new_tokens = 128, 
                            neighbours= n, use_openai_embedding= use_openai_embedding, caption_type=c)
    