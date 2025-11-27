#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from goldfish_lv_json import GoldFish_LV 
import time
from goldfish.paths import CHECKPOINTS_DIR, TEST_CONFIGS_DIR, VIDEOS_DIR

DEFAULT_CFG_PATH = TEST_CONFIGS_DIR / "llama2_test_config.yaml"
DEFAULT_CKPT_PATH = CHECKPOINTS_DIR / "video_mistral_checkpoint_last.pth"
DEFAULT_FOLDER_PATH = VIDEOS_DIR / "season_1"
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", default=str(DEFAULT_CFG_PATH))
    parser.add_argument("--neighbours", type=int, default=3)
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--add_subtitles", action='store_true')
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_openai_embedding",type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for short video clips")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--video_path", type=str,default="path for video.mp4", help="Path to the video file or youtube url")
    parser.add_argument(
        "--folder_path",
        type=str,
        default=str(DEFAULT_FOLDER_PATH),
        help="Path to the folder containing episode .mp4 files",
    )
    parser.add_argument("--question", type=str, default="Why rachel is wearing a wedding dress?")
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()

def download_video(youtube_url):
    processed_video_path = goldfish_lv.process_video_url(youtube_url)
    return processed_video_path

def process_video(video_path, has_subtitles, instruction="",number_of_neighbours=-1):
    result = goldfish_lv.inference(video_path, has_subtitles, instruction,number_of_neighbours)
    #pred = result["pred"]
    return result # pred

def return_video_path(youtube_url):
    video_id = youtube_url.split("https://www.youtube.com/watch?v=")[-1].split('&')[0]
    if video_id:
        return os.path.join("workspace", "tmp", f"{video_id}.mp4")
    else:
        raise ValueError("Invalid YouTube URL provided.")
args=get_arguments()
if __name__ == "__main__":
    t1=time.time()
    print("using openai: ", args.use_openai_embedding)
    goldfish_lv = GoldFish_LV(args)
    t2=time.time()
    print("Time taken to load model: ", t2-t1)
    video_folder = Path(args.folder_path)
    video_paths = sorted(video_folder.glob("*.mp4"))
    #episodes = ["episode_20"]
    for video_path in video_paths:
        #if any(ep in video_path.lower() for ep in episodes):    
        print(f'Processing {video_path}')
        processed_video_path = goldfish_lv.process_video_url(str(video_path))
        pred=process_video(processed_video_path, args.add_subtitles, args.question,args.neighbours)      
        #print("Question answer: ", pred)
        print("Time taken for inference: ", time.time()-t2)
