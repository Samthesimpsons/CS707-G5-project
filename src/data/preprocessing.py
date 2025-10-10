"""Data preprocessing pipeline for CS707 G5 Project.

This module combines download, extraction, and preprocessing functionality
into a single unified pipeline for the Friends TV show dataset.
"""

import os
import json
import re
import shutil
import subprocess
import tarfile
import requests  # type: ignore[import-untyped]
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional, List, Any, Dict, cast, Tuple

from openai import OpenAI  # type: ignore[import-not-found]
from bs4 import BeautifulSoup  # type: ignore[import-not-found]
from dotenv import load_dotenv  # type: ignore[import-not-found]
from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]


LLM_MODEL = "gpt-4o-mini"


@dataclass
class DownloadConfig:
    """Configuration for dataset download.

    Attributes:
        output_dir: Directory where the dataset will be saved.
        hf_token: Hugging Face API token for authentication.
        repo_id: Repository ID on Hugging Face Hub.
    """

    output_dir: Path = Path("./data")
    hf_token: Optional[str] = None
    repo_id: str = "Vision-CAIR/TVQA-Long"


VIDEO_FILES = [
    "tvqa-long-videos/archive.tar.gz.aa",
    "tvqa-long-videos/archive.tar.gz.ab",
    "tvqa-long-videos/archive.tar.gz.ac",
    "tvqa-long-videos/archive.tar.gz.ad",
    "tvqa-long-videos/archive.tar.gz.ae",
    "tvqa-long-videos/archive.tar.gz.af",
    "tvqa-long-videos/archive.tar.gz.ag",
    "tvqa-long-videos/archive.tar.gz.ah",
    "tvqa-long-videos/archive.tar.gz.ai",
    "tvqa-long-videos/archive.tar.gz.aj",
    "tvqa-long-videos/archive.tar.gz.ak",
]

SUBTITLE_FILE = "tvqa-long-annotations/tvqa_preprocessed_subtitles.json"

FRIENDS_SCRIPTS_REPO = "edersoncorbari/friends-scripts"
FRIENDS_SCRIPTS_BASE_URL = (
    "https://api.github.com/repos/edersoncorbari/friends-scripts/contents/season"
)


def load_environment_variables() -> None:
    """Load environment variables from .env file.

    This function loads the .env file from the project root directory and makes
    the environment variables available via os.environ.

    Raises:
        FileNotFoundError: If .env file is not found.
    """
    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError(
            ".env file not found. Please create one using .env.example as a template."
        )

    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from {env_path.absolute()}")


def get_hf_token() -> Optional[str]:
    """Get Hugging Face token from environment variables.

    Returns:
        Hugging Face token string or None if not set.
    """
    token = os.getenv("HUGGING_FACE_TOKEN")
    if token and token != "your_hf_token_here":
        print("Hugging Face token loaded")
        return token
    else:
        print(
            "Warning: No Hugging Face token found. Download may fail for gated repos."
        )
        return None


def download_file(
    repo_id: str, filename: str, local_dir: Path, token: Optional[str] = None
) -> Path:
    """Download a single file from Hugging Face Hub.

    Args:
        repo_id: Repository ID on Hugging Face Hub.
        filename: Path to file within the repository.
        local_dir: Local directory to save the file.
        token: Hugging Face API token.

    Returns:
        Path to the downloaded file.

    Raises:
        Exception: If download fails.
    """
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
            repo_type="dataset",
        )
        return Path(downloaded_path)
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        raise


def get_friends_script_files() -> List[str]:
    """Get list of all Friends script HTML files from GitHub.

    Returns:
        List of script file names.

    Raises:
        Exception: If GitHub API request fails.
    """
    try:
        response = requests.get(FRIENDS_SCRIPTS_BASE_URL)
        response.raise_for_status()
        files = response.json()
        script_files = [f["name"] for f in files if f["name"].endswith(".html")]
        return sorted(script_files)
    except Exception as e:
        print(f"Error fetching Friends scripts list: {e}")
        raise


def download_friends_script(script_file: str, output_dir: Path) -> Path:
    """Download a single Friends script HTML file from GitHub.

    Args:
        script_file: Name of the script file.
        output_dir: Directory to save the script file.

    Returns:
        Path to the downloaded script file.

    Raises:
        Exception: If download fails.
    """
    raw_url = f"https://raw.githubusercontent.com/{FRIENDS_SCRIPTS_REPO}/master/season/{script_file}"
    output_path = output_dir / script_file

    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        output_path.write_text(response.text, encoding="utf-8")
        return output_path
    except Exception as e:
        print(f"Error downloading {script_file}: {e}")
        raise


def download_edersoncorbari_friends_scripts(output_dir: Path) -> None:
    """Download all Friends scripts from edersoncorbari GitHub.

    Args:
        output_dir: Base directory to save the scripts.

    Raises:
        Exception: If download fails.
    """
    print(f"\n{'='*80}")
    print("Downloading Friends Scripts from edersoncorbari GitHub")
    print(f"{'='*80}\n")

    scripts_dir = output_dir / "edersoncorbari_subtitles"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_files = get_friends_script_files()
    total_scripts = len(script_files)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_script = {
            executor.submit(
                download_friends_script, script_file, scripts_dir
            ): script_file
            for script_file in script_files
        }
        failed_downloads = []
        for i, future in enumerate(as_completed(future_to_script), 1):
            script_file = future_to_script[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{i}/{total_scripts}] Failed: {script_file} - {e}")
                failed_downloads.append(script_file)

    print(f"Friends scripts downloaded to: {scripts_dir}")


def download_tvqa_dataset(config: DownloadConfig) -> None:
    """Download TVQA-Long dataset files.

    Args:
        config: Configuration object containing download settings.

    Raises:
        Exception: If any download fails.
    """
    print(f"\n{'='*80}")
    print(f"Downloading Dataset from repository: {config.repo_id}")
    print(f"{'='*80}\n")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    total_files = len(VIDEO_FILES)
    current_file = 0

    print("Downloading Subtitles (all shows - will filter Friends during extraction)")
    download_file(config.repo_id, SUBTITLE_FILE, config.output_dir, config.hf_token)

    print(
        "Downloading Video Archives in parallel (will filter to Friends during extraction)"
    )
    print("This will take a while - ~50GB total")

    with ThreadPoolExecutor(max_workers=len(VIDEO_FILES)) as executor:
        future_to_video = {
            executor.submit(
                download_file,
                config.repo_id,
                video_file,
                config.output_dir,
                config.hf_token,
            ): video_file
            for video_file in VIDEO_FILES
        }

        for future in as_completed(future_to_video):
            current_file += 1
            video_file = future_to_video[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{current_file}/{total_files}] Failed: {video_file} - {e}")
                raise

    print(f"TVQA-Long Dataset downloaded to: {config.output_dir}")


def combine_split_archives(data_dir: Path) -> Path:
    """Combine split video archives into a single tar.gz file.

    Args:
        data_dir: Directory containing the split archives.

    Returns:
        Path to the combined archive file.

    Raises:
        FileNotFoundError: If split archives are not found.
    """
    print(f"\n{'='*80}")
    print("Step: Combining Split Video Archives")
    print(f"{'='*80}\n")

    video_dir = data_dir / "tvqa-long-videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    archive_parts = ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak"]
    missing_parts = []
    for part in archive_parts:
        part_path = video_dir / f"archive.tar.gz.{part}"
        if not part_path.exists():
            missing_parts.append(part)

    if missing_parts:
        raise FileNotFoundError(
            f"Missing archive parts: {', '.join(missing_parts)}\n"
            "Please re-run download step first."
        )

    combined_archive = video_dir / "archive.tar.gz"
    with open(combined_archive, "wb") as outfile:
        for i, part in enumerate(archive_parts, 1):
            part_path = video_dir / f"archive.tar.gz.{part}"
            with open(part_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print(f"\nCombined archive created: {combined_archive}")
    print(f"Size: {combined_archive.stat().st_size / (1024**3):.2f} GB")

    print("\nCleaning up split archive files...")
    for i, part in enumerate(archive_parts, 1):
        part_path = video_dir / f"archive.tar.gz.{part}"
        part_path.unlink()
    print("Split archive files deleted.")

    return combined_archive


def extract_video_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract Friends videos only from the combined archive.

    Args:
        archive_path: Path to the combined tar.gz archive.
        output_dir: Directory to extract videos to.

    Returns:
        Path to the extracted Friends videos directory.

    Raises:
        FileNotFoundError: If archive is not found.
    """
    print(f"\n{'='*80}")
    print("Step: Extracting Friends Videos Only")
    print(f"{'='*80}\n")

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    friends_videos_dir = output_dir / "videos"

    if friends_videos_dir.exists() and list(friends_videos_dir.iterdir()):
        print(
            f"Friends videos directory already exists and is not empty: {friends_videos_dir}"
        )
        print("Skipping extraction step.")
        return friends_videos_dir

    friends_videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting Friends videos to: {friends_videos_dir}")
    print("Filtering for Friends videos only...")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()

            friends_members = [
                m
                for m in members
                if "/friends/" in m.name.lower()
                or m.name.lower().startswith("friends/")
            ]
            total_friends = len(friends_members)

            if total_friends == 0:
                print("Warning: No Friends videos found in archive!")
                return friends_videos_dir

            print(
                f"Found {total_friends} Friends video files out of {len(members)} total files"
            )
            print("Extracting Friends videos and reorganizing by season...")

            for i, member in enumerate(friends_members, 1):
                if i % 10 == 0 or i == total_friends:
                    print(f"Progress: {i}/{total_friends} files", end="\r")

                path_parts = member.name.split("/")
                season_part = None
                for part in path_parts:
                    if "season" in part.lower():
                        season_part = part
                        break

                if season_part and member.isfile():
                    season_dir = friends_videos_dir / season_part
                    season_dir.mkdir(parents=True, exist_ok=True)

                    filename = Path(member.name).name
                    dest_path = season_dir / filename

                    file_obj = tar.extractfile(member)
                    if file_obj:
                        dest_path.write_bytes(file_obj.read())
            print()

        print(f"\nFriends videos extracted to: {friends_videos_dir}")

        video_files = list(friends_videos_dir.rglob("*.mp4")) + list(
            friends_videos_dir.rglob("*.mkv")
        )
        print(f"Total Friends video files: {len(video_files)}")

        print("\nCleaning up combined archive file...")
        archive_path.unlink()
        print(f"Deleted: {archive_path}")

        video_archive_dir = output_dir / "tvqa-long-videos"
        if video_archive_dir.exists():
            print("\nCleaning up tvqa-long-videos directory...")
            shutil.rmtree(video_archive_dir)
            print(f"Deleted: {video_archive_dir}")

    except Exception as e:
        print(f"\nError extracting archive: {e}")
        raise

    return friends_videos_dir


def extract_subtitles_from_preprocessed(
    data_dir: Path, keyword: str = "friends"
) -> Path:
    """Extract Friends subtitles from tvqa_preprocessed_subtitles.

    Args:
        data_dir: Directory containing the preprocessed subtitles.
        keyword: Keyword to filter for (default: "friends").

    Returns:
        Path to the extracted Friends subtitles JSON file.

    Raises:
        FileNotFoundError: If preprocessed subtitles file is not found.
    """
    print(f"\n{'='*80}")
    print("Step: Extracting Friends Subtitle From TVQA Preprocessed")
    print(f"{'='*80}\n")

    subtitles_dir = "tvqa-long-annotations"
    tvqa_preprocessed_subs_file = (
        data_dir / subtitles_dir / "tvqa_preprocessed_subtitles.json"
    )
    if not tvqa_preprocessed_subs_file.exists():
        raise FileNotFoundError(
            f"Subtitle file not found: {tvqa_preprocessed_subs_file}"
        )

    with open(tvqa_preprocessed_subs_file, "r") as subs_file:
        processed_subs = json.load(subs_file)
        subs_file.close()

    processed_subs_friends = [
        sub for sub in processed_subs if keyword.lower() in sub["vid_name"].lower()
    ]
    processed_subs_friends = sorted(processed_subs_friends, key=lambda x: x["vid_name"])

    for sub_item in processed_subs_friends:
        if sub_item["sub"]:
            clip_start = sub_item["sub"][0]["start"]
            clip_end = sub_item["sub"][-1]["end"]
            sub_item["clip_timings"] = {"start": clip_start, "end": clip_end}

        subjects = set()
        for dialogue in sub_item["sub"]:
            text = dialogue["text"].strip()
            match = re.match(r"^([A-Za-z]+)\s*:", text)
            if match:
                subjects.add(match.group(1))
        sub_item["subjects"] = list(subjects)

    output_dir = data_dir / "tvqa_preprocessed_friends_subtitles.json"
    with open(output_dir, "w") as output:
        json.dump(processed_subs_friends, output, indent=4)

    print(f"Friends preprocessed subs saved to {output_dir}")

    print("\nCleaning up tvqa-long-annotations path...")
    path_to_clean = data_dir / subtitles_dir
    shutil.rmtree(path_to_clean)
    print(f"Deleted: {path_to_clean}")

    return output_dir


def parse_html_script(html_path: Path) -> Dict[str, Any]:
    """Parse HTML script file to extract scenes and dialogue.

    Handles multiple HTML format variations found across Friends episodes:

    Format 1: Standard with tags (e.g., 0101.html)
        <p><b>Speaker:</b> dialogue text</p>
        <p><strong>Speaker</strong>: dialogue text</p>

    Format 2: All caps without tags (e.g., 0203.html)
        <p>SPEAKER: dialogue text</p>

    Format 3: Multiple dialogues per paragraph with <br> (e.g., 0204.html)
        <p>SPEAKER: dialogue<br>SPEAKER: dialogue<br>...</p>

    Format 4: Malformed HTML with dialogue outside <p> tags (e.g., 0206.html)
        - Dialogue extracted from body text when paragraphs fail

    Format 5: Speaker and dialogue on separate lines (e.g., 0302.html)
        <b><p>Speaker:</b> dialogue on next line</p>

    Format 6: Dialogue split by <br> within tags (e.g., 0915.html)
        <p><B>Speaker</B>: text<br><B>Speaker</B>: text<br>...</p>

    All speaker names are normalized to uppercase for consistency.
    Metadata lines (written by, transcribed by, etc.) are filtered out.

    Args:
        html_path: Path to HTML script file.

    Returns:
        Dictionary with episode info, scenes, and dialogue:
        {
            "episode": "0101",
            "title": "...",
            "scenes": [
                {
                    "scene_description": "...",
                    "dialogue": [
                        {"speaker": "MONICA", "text": "..."},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text() if title_tag else ""

    episode_num = html_path.stem

    scenes: List[Dict[str, Any]] = []
    current_scene: Dict[str, Any] = {
        "scene_description": "Episode dialogue",
        "dialogue": [],
    }

    paragraphs = soup.find_all("p")

    for p in paragraphs:
        if p.find("br"):
            html_str = str(p)
            br_parts = re.split(r"<br\s*/?>", html_str, flags=re.IGNORECASE)

            lines = []
            for part in br_parts:
                part_soup = BeautifulSoup(part, "html.parser")
                part_text = part_soup.get_text(strip=True)
                if part_text:
                    lines.append(part_text)
        else:
            text = p.get_text(strip=True)
            lines = [text] if text else []

        for text in lines:
            is_scene_marker = (
                text.startswith("[Scene:")
                or text.startswith("[Opening")
                or text.startswith("[Closing")
            )

            if is_scene_marker:
                if current_scene and current_scene["dialogue"]:
                    scenes.append(current_scene)

                current_scene = cast(
                    Dict[str, Any], {"scene_description": text, "dialogue": []}
                )
                continue

            if ":" not in text:
                continue

            match = re.match(r"^([A-Za-z\s.]+?):\s*(.+)$", text)
            if match:
                speaker = match.group(1).strip()
                dialogue_text = match.group(2).strip()

                speaker = re.sub(r"<[^>]+>", "", speaker)
                speaker = speaker.upper()

                skip_keywords = [
                    "written",
                    "transcribed",
                    "originally",
                    "minor",
                    "adjustment",
                    "story",
                    "teleplay",
                    "directed",
                ]
                if any(keyword in speaker.lower() for keyword in skip_keywords):
                    continue

                max_speaker_name_length = 30
                if len(speaker) > max_speaker_name_length:
                    continue

                current_scene["dialogue"].append(
                    {"speaker": speaker, "text": dialogue_text}
                )

    if current_scene and current_scene["dialogue"]:
        scenes.append(current_scene)

    total_dialogue = sum(len(scene["dialogue"]) for scene in scenes)
    if not scenes or total_dialogue == 0:
        body = soup.find("body")
        if body:
            all_text = body.get_text(separator="\n")
            lines = [line.strip() for line in all_text.split("\n") if line.strip()]

            fallback_scene: Dict[str, Any] = {
                "scene_description": "Episode dialogue",
                "dialogue": [],
            }

            i = 0
            while i < len(lines):
                text = lines[i]

                is_scene_marker = (
                    text.startswith("[Scene:")
                    or text.startswith("[Opening")
                    or text.startswith("[Closing")
                )

                if is_scene_marker:
                    if fallback_scene["dialogue"]:
                        scenes.append(fallback_scene)
                    fallback_scene = cast(
                        Dict[str, Any], {"scene_description": text, "dialogue": []}
                    )
                    i += 1
                    continue

                if text.endswith(":") and i + 1 < len(lines):
                    speaker = text[:-1].strip()

                    speaker = re.sub(r"<[^>]+>", "", speaker)
                    skip_keywords = [
                        "written",
                        "transcribed",
                        "originally",
                        "minor",
                        "adjustment",
                        "story",
                        "teleplay",
                        "directed",
                    ]

                    max_speaker_name_length = 30
                    is_valid_speaker_name = re.match(r"^[A-Za-z\s.&]+$", speaker)

                    if (
                        len(speaker) <= max_speaker_name_length
                        and not any(
                            keyword in speaker.lower() for keyword in skip_keywords
                        )
                        and is_valid_speaker_name
                    ):

                        dialogue_parts = []
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j]
                            is_next_speaker = next_line.endswith(":") and re.match(
                                r"^[A-Za-z\s.&]+:$", next_line
                            )
                            is_next_scene = (
                                next_line.startswith("[Scene:")
                                or next_line.startswith("[Opening")
                                or next_line.startswith("[Closing")
                            )

                            if is_next_speaker or is_next_scene:
                                break
                            dialogue_parts.append(next_line)
                            j += 1

                        if dialogue_parts:
                            dialogue_text = " ".join(dialogue_parts)
                            fallback_scene["dialogue"].append(
                                {"speaker": speaker.upper(), "text": dialogue_text}
                            )
                        i = j
                        continue

                if ":" in text:
                    match = re.match(r"^([A-Za-z\s.]+?):\s*(.+)$", text)
                    if match:
                        speaker = match.group(1).strip()
                        dialogue_text = match.group(2).strip()

                        speaker = re.sub(r"<[^>]+>", "", speaker)
                        speaker = speaker.upper()

                        skip_keywords = [
                            "written",
                            "transcribed",
                            "originally",
                            "minor",
                            "adjustment",
                            "story",
                            "teleplay",
                            "directed",
                        ]
                        if any(keyword in speaker.lower() for keyword in skip_keywords):
                            i += 1
                            continue

                        if len(speaker) > 30:
                            i += 1
                            continue

                        fallback_scene["dialogue"].append(
                            {"speaker": speaker, "text": dialogue_text}
                        )

                i += 1

            if fallback_scene["dialogue"]:
                scenes.append(fallback_scene)

            if scenes and total_dialogue == 0:
                scenes = [s for s in scenes if len(s["dialogue"]) > 0]

    return {"episode": episode_num, "title": title, "scenes": scenes}


def convert_all_html_scripts(input_dir: Path, output_dir: Path) -> None:
    """Convert all HTML scripts to JSON format.

    Args:
        input_dir: Directory containing HTML script files.
        output_dir: Directory to save JSON files.
    """
    print(f"\n{'='*80}")
    print("Converting HTML Scripts to JSON (Edersoncorbari)")
    print(f"{'='*80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(input_dir.glob("*.html"))

    episode_files = [f for f in html_files if re.match(r"^\d{4}", f.stem)]

    print(f"Found {len(episode_files)} episode scripts")

    for i, html_file in enumerate(episode_files, 1):
        try:
            if i % 10 == 0 or i == len(episode_files):
                print(f"Progress: {i}/{len(episode_files)}", end="\r")

            data = parse_html_script(html_file)

            output_file = output_dir / f"{html_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {html_file.name}: {e}")

    print(f"\n\nHTML scripts converted: {len(episode_files)} files")
    print(f"Output directory: {output_dir}")


def seconds_to_ffmpeg_time(seconds: float) -> str:
    """Convert seconds to ffmpeg timestamp format.

    Args:
        seconds: Time in seconds.

    Returns:
        Timestamp in format "HH:MM:SS.mmm".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_vid_name(vid_name: str) -> Dict[str, Any]:
    """Parse video name to extract season, episode, segment, and clip info.

    Args:
        vid_name: Video name in format "friends_s01e01_seg02_clip_01".

    Returns:
        Dictionary with season, episode, segment, and clip numbers.
    """
    # Pattern: friends_s{season}e{episode}_seg{segment}_clip_{clip}
    match = re.match(r"friends_s(\d+)e(\d+)_seg(\d+)_clip_(\d+)", vid_name)
    if match:
        season, episode, segment, clip = match.groups()
        return {
            "season": int(season),
            "episode": int(episode),
            "segment": int(segment),
            "clip": int(clip),
            "episode_id": f"{int(season):02d}{int(episode):02d}",
        }
    return {}


def find_video_file_for_clip(episode_id: str, videos_dir: Path) -> Path:
    """Find video file for given episode.

    Args:
        episode_id: Episode identifier in format "0101".
        videos_dir: Base videos directory.

    Returns:
        Path to video file.
    """
    season_num = f"season_{int(episode_id[:2])}"
    episode_num = f"episode_{int(episode_id[2:])}.mp4"
    return videos_dir / season_num / episode_num


def extract_video_clip_from_tvqa(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
) -> bool:
    """Extract video clip using ffmpeg.

    Args:
        video_path: Path to source video file.
        start_seconds: Start time in seconds.
        end_seconds: End time in seconds.
        output_path: Path for output clip.

    Returns:
        True if successful, False otherwise.
    """
    try:
        duration = end_seconds - start_seconds
        start_ffmpeg = seconds_to_ffmpeg_time(start_seconds)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            start_ffmpeg,
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            str(output_path),
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode('utf-8', errors='ignore')[:200]}")
        return False
    except Exception as e:
        print(f"Extraction error: {e}")
        return False


def extract_clips_from_tvqa_combined(
    tvqa_combined_path: Path,
    videos_dir: Path,
    output_clips_dir: Path,
) -> None:
    """Extract video clips from tvqa_combined.json.

    Args:
        tvqa_combined_path: Path to tvqa_combined.json file.
        videos_dir: Directory containing video files.
        output_clips_dir: Directory to save extracted clips.
    """
    print(f"\n{'='*80}")
    print("Extracting Video Clips from TVQA Combined Data")
    print(f"{'='*80}\n")

    if not tvqa_combined_path.exists():
        raise FileNotFoundError(f"TVQA combined file not found: {tvqa_combined_path}")

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    with open(tvqa_combined_path, "r", encoding="utf-8") as f:
        tvqa_data = json.load(f)

    total_clips = len(tvqa_data)
    clips_extracted = 0
    clips_failed = 0
    clips_skipped = 0

    print(f"Total clips to extract: {total_clips}")

    for i, clip_data in enumerate(tvqa_data, 1):
        vid_name = clip_data.get("vid_name", "")
        clip_timings = clip_data.get("clip_timings", {})

        if not clip_timings or "start" not in clip_timings or "end" not in clip_timings:
            clips_skipped += 1
            continue

        parsed = parse_vid_name(vid_name)
        if not parsed:
            clips_skipped += 1
            continue

        episode_id = parsed["episode_id"]
        video_path = find_video_file_for_clip(episode_id, videos_dir)

        if not video_path:
            clips_skipped += 1
            continue

        start_seconds = clip_timings["start"]
        end_seconds = clip_timings["end"]

        clip_filename = f"{vid_name}.mp4"
        clip_output_path = output_clips_dir / episode_id / clip_filename

        if clip_output_path.exists():
            clips_extracted += 1
            continue

        success = extract_video_clip_from_tvqa(
            video_path, start_seconds, end_seconds, clip_output_path
        )

        if success:
            clips_extracted += 1
        else:
            clips_failed += 1
            print(f"[{i}/{total_clips}] Failed {vid_name}")

    print("Clip Extraction Summary:")
    print(f"Total clips: {total_clips}")
    print(f"Successfully extracted: {clips_extracted}")
    print(f"Failed: {clips_failed}")
    print(f"Skipped (no video/timing): {clips_skipped}")
    print(f"Output directory: {output_clips_dir}\n")

    return None


def prepare_batch_requests_from_tvqa(
    tvqa_file: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Prepare batch requests from TVQA combined format.

    Args:
        tvqa_file: Path to the TVQA combined JSON file.

    Returns:
        Tuple of (batch_requests_list, metadata_dict) where metadata contains
        information needed to reconstruct memory tuples later.
    """
    batch_requests = []
    metadata = {}

    print(f"Loading TVQA data from: {tvqa_file}")
    with open(tvqa_file, "r", encoding="utf-8") as f:
        tvqa_data = json.load(f)

    print(f"Preparing batch requests for {len(tvqa_data)} clips...")

    for idx, clip_data in enumerate(tvqa_data):
        vid_name = clip_data.get("vid_name", "")
        subtitles = clip_data.get("sub", [])
        clip_timings = clip_data.get("clip_timings", {"start": 0, "end": 0})
        subjects = clip_data.get("subjects", [])
        location = ""  # To fill in from manual process

        dialogue_entries = []
        for sub in subtitles:
            text = sub.get("text", "").strip()
            start_time = sub.get("start", 0)
            end_time = sub.get("end", 0)

            if ":" in text:
                parts = text.split(":", 1)
                speaker = parts[0].strip()
                dialogue_text = parts[1].strip() if len(parts) > 1 else ""
            else:
                speaker = "UNKNOWN"
                dialogue_text = text

            if dialogue_text:
                dialogue_entries.append(
                    {
                        "speaker": speaker,
                        "text": dialogue_text,
                        "start": start_time,
                        "end": end_time,
                    }
                )

        dialogue_text = "\n".join(
            f"{entry['speaker']}: {entry['text']}" for entry in dialogue_entries
        )

        schema_block = """
        {
            "detailed_description": "string",
            "summary_verb_noun": "string",
            "flashback_events": [
                "string"
            ]
        }
        """.strip()

        example_block = """
        {
            "detailed_description": "Ross, Rachel, and Joey discuss the aftermath of Chandler's awkward joke while Monica prepares dinner. The atmosphere is light-hearted yet tense.",
            "summary_verb_noun": "Discuss Chandler's Joke",
            "flashback_events": ["Ross mentions his failed wedding in London"]
        }
        """.strip()

        llm_prompt = dedent(
            """\
        You are given a snippet from an episode of a TV show.
        Analyze the snippet and extract structured information in strict JSON format.

        Here is context of a snippet of an Episode of a TV Show
        Scene Location:
        {location}

        Characters Present:
        {subjects}

        Dialogue:
        {dialogue}

        Your task:
        Extract and return the following fields only in valid JSON format (no additional text, comments, or explanations).

        Output Schema:
        {schema_block}

        Field Guidelines:
        - "detailed_description": Write a full narrative of what is happening, including key actions, tone, and emotional context.
        - "summary_verb_noun": Summarize the main event using a Verb + Noun structure (e.g., "Celebrate Ross's Birthday").
        - "flashback_events": List all flashback events or references to past occurrences. If none, return an empty array ([]).

        Example Output:
        {example_block}
        """
        ).format(
            location=location,
            subjects=", ".join(subjects),
            dialogue=dialogue_text,
            schema_block=schema_block,
            example_block=example_block,
        )

        batch_request = {
            "custom_id": vid_name,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that describes TV show scenes concisely.",
                    },
                    {"role": "user", "content": llm_prompt},
                ],
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "scene_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "detailed_description": {
                                    "type": "string",
                                    "description": "Full narrative of what is happening",
                                },
                                "summary_verb_noun": {
                                    "type": "string",
                                    "description": "Main event using Verb + Noun structure",
                                },
                                "flashback_events": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of flashback events or references to past occurrences",
                                },
                            },
                            "required": [
                                "detailed_description",
                                "summary_verb_noun",
                                "flashback_events",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        }

        batch_requests.append(batch_request)

        duration_seconds = clip_timings.get("end", 0) - clip_timings.get("start", 0)

        metadata[vid_name] = {
            "vid_name": vid_name,
            "subjects": subjects,
            "start_time": clip_timings.get("start", 0),
            "end_time": clip_timings.get("end", 0),
            "duration_seconds": duration_seconds,
            "location": location,
        }

    print(f"Prepared {len(batch_requests)} batch requests\n")
    return batch_requests, metadata


def submit_batch_job(
    client: OpenAI, batch_requests: List[Dict[str, Any]], temp_dir: Path
) -> str:
    """Submit batch job to OpenAI.

    Args:
        client: OpenAI client instance.
        batch_requests: List of batch request dictionaries.
        temp_dir: Directory to store temporary JSONL file.

    Returns:
        Batch job ID.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    batch_file_path = temp_dir / "batch_requests_tvqa.jsonl"

    with open(batch_file_path, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    print(f"Created batch file: {batch_file_path}")

    with open(batch_file_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    print(f"Uploaded batch file: {batch_file.id}")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"Created batch job: {batch_job.id}")
    print(f"Status: {batch_job.status}\n")

    return batch_job.id


def save_batch_state(
    batch_job_id: str, metadata: Dict[str, Dict[str, Any]], state_file: Path
) -> None:
    """Save batch job state to file for later resumption.

    Args:
        batch_job_id: Batch job ID.
        metadata: Metadata dictionary with clip information.
        state_file: Path to save the state file.
    """
    state_data = {
        "batch_job_id": batch_job_id,
        "metadata": metadata,
    }

    state_file.parent.mkdir(parents=True, exist_ok=True)

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2, ensure_ascii=False)

    print(f"Saved batch job state to: {state_file}\n")


def create_batch_job(tvqa_combined_path: Path) -> None:
    """Prepare and submit batch job for event descriptions.

    Args:
        tvqa_combined_path: Path to tvqa_combined.json file.
    """
    print(f"\n{'='*80}")
    print("TVQA Batch Processing using OpenAI Batch API - Stage 1")
    print(f"{'='*80}\n")

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    batch_requests, metadata = prepare_batch_requests_from_tvqa(tvqa_combined_path)

    temp_dir = Path("./data/temp")
    batch_job_id = submit_batch_job(client, batch_requests, temp_dir)

    state_file = temp_dir / "batch_state_tvqa.json"
    save_batch_state(batch_job_id, metadata, state_file)

    print("STAGE 1 COMPLETE:")
    print(f"Batch job ID: {batch_job_id}")
    print(f"Total clips to process: {len(metadata)}")
    print(f"State saved to: {state_file}")
    print(
        "\nNote: Batch jobs can take up to 24 hours but usually complete much faster."
    )
    print("You can check status at: https://platform.openai.com/batches\n")


def preprocessing_pipeline() -> None:
    """
    Main data preprocessing pipeline.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    data_dir = Path("./data")

    try:
        load_environment_variables()

        hf_token = get_hf_token()

        config = DownloadConfig(
            output_dir=data_dir,
            hf_token=hf_token,
        )

        download_edersoncorbari_friends_scripts(config.output_dir)

        download_tvqa_dataset(config)

        combined_archive = combine_split_archives(data_dir)

        extract_video_archive(combined_archive, data_dir)

        extract_subtitles_from_preprocessed(data_dir)

        html_input_dir = data_dir / "edersoncorbari_subtitles"
        html_output_dir = data_dir / "subtitles" / "edersoncorbari"

        if html_input_dir.exists():
            convert_all_html_scripts(html_input_dir, html_output_dir)

            shutil.rmtree(html_input_dir)
        else:
            print(f"Warning: HTML scripts directory not found: {html_input_dir}")

        json_input_dir = data_dir / "tvqa_preprocessed_friends_subtitles.json"
        json_output_dir = data_dir / "subtitles" / "tvqa_combined.json"

        if json_input_dir.exists():
            shutil.move(str(json_input_dir), str(json_output_dir))
        else:
            print(f"Warning: TVQA subtitles directory not found: {json_input_dir}")

        tvqa_combined_path = data_dir / "subtitles" / "tvqa_combined.json"
        videos_dir = data_dir / "videos"
        output_clips_dir = data_dir / "clips"

        extract_clips_from_tvqa_combined(
            tvqa_combined_path, videos_dir, output_clips_dir
        )

        print(f"\n{'='*80}")
        print("MANUAL EDITING REQUIRED")
        print(f"{'='*80}\n")
        print(
            "The clips have been extracted. Now you need to manually edit the locations."
        )
        print(f"\nFile to edit: {tvqa_combined_path}")
        print("\nInstructions:")
        print("1. Open the tvqa_combined.json file")
        print("2. Manually add/edit the 'location' field for each clip entry")
        print("3. Save the updated file")
        print("4. Replace the original file with your edited version")
        print("5. Press ENTER to continue to batch job creation...\n")

        input("Press ENTER when you're ready to proceed: ")

        print("\nResuming pipeline...\n")

        create_batch_job(json_output_dir)

    except Exception as e:
        print(f"\nError in data pipeline: {e}")
        exit(1)
