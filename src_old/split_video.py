import time
import json

from moviepy import VideoFileClip
from dataclasses import dataclass
from pathlib import Path


DATA = Path("./data/")
EPISODIC_TUPLES_DIR = DATA / "annotated_tuples"
VIDEO_INPUT_DIR = DATA / "video" / "season_1"
VIDEO_OUTPUT_DIR = DATA / "video_clip"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class EpisodicTuple:
    episode_file: str 
    clip_start: str
    clip_end: str 
    clip_name: str


def convert_timestamp_seconds(tstamp: str):
    # Accepted timestamp = hh:mm:ss - """Convert hh:mm:ss to total seconds."""
    h, m, s = map(int, tstamp.split(":"))
    total_secs = h * 3600 + m * 60 + s
    return total_secs


def fetch_params_from_annotations(episode_file: str, scene: dict):
    return EpisodicTuple(
        episode_file=episode_file,
        clip_start=scene.get("clip_start", ""),
        clip_end=scene.get("clip_end", ""),
        clip_name=scene.get("context_label", "")
    )


def split_video(input_vid_path: Path, output_vid_path: Path, params: EpisodicTuple):
    if output_vid_path.exists():
        print(f"Skipping {output_vid_path} - Clip Already Exists")
        return output_vid_path
    
    # Load the video
    video = VideoFileClip(str(input_vid_path))
    t1 = convert_timestamp_seconds(params.clip_start)
    t2 = convert_timestamp_seconds(params.clip_end)
    
    # Create the subclip
    video_clip = video.subclipped(t1, t2)
    video_clip.write_videofile(str(output_vid_path), codec="libx264")
    
    print(f"splitting of video completed - {output_vid_path}")
    video.close()


def process_videos_from_annotations(
    files_input_dir: Path,
    video_input_dir: Path,
    video_output_dir: Path
):
    json_files = list(files_input_dir.glob("*.json"))
    print(json_files)
    ### Read from file containing annotated json tuple files
    for json_path in json_files:
        with open(json_path, 'r') as file:
            annotated_tuple = json.load(file)
            
        episode_string = annotated_tuple.get("episode", "0000")
        episode_num = int(episode_string[-2:])
        episode_file = f"episode_{episode_num}.mp4"

        input_video_path = video_input_dir / episode_file

        print(f"\n{'='*80}")
        print(f"Splitting: {input_video_path}")
        print(f"{'='*80}\n")
        
        for scene in annotated_tuple.get("scenes", []):
            ### Run fetch_params_from_annotations
            params = fetch_params_from_annotations(episode_file, scene)

            ### Run split video
            try:
                if params.clip_name:
                    clip_name = "_".join(params.clip_name.split(" ")).lower().replace("/", "_")
                    episode_output_path = video_output_dir / f"{clip_name}.mp4"
                    split_video(input_video_path, episode_output_path, params)
                else:
                    print(f"Unknown clip name {params.clip_name} for {json_path}")
                    continue
            except Exception as e:
                print(f"Clip:{clip_name}-{e}")


if __name__ == "__main__":
    process_videos_from_annotations(
        files_input_dir = EPISODIC_TUPLES_DIR,
        video_input_dir = VIDEO_INPUT_DIR,
        video_output_dir = VIDEO_OUTPUT_DIR
    )






    
    