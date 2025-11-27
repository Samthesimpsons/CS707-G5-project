import os
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)

from goldfish.paths import CHECKPOINTS_DIR  # ty: ignore


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

FRIENDS_SCRIPTS_REPO = "edersoncorbari/friends-scripts"

FRIENDS_SCRIPTS_BASE_URL = (
    "https://api.github.com/repos/edersoncorbari/friends-scripts/contents/season"
)


@dataclass
class DownloadConfig:
    """Options for pulling the TVQA-Long dataset from Hugging Face.

    Attributes:
        output_dir: Destination directory for the downloaded dataset.
        hf_token: Hugging Face API token used for gated assets.
        repo_id: Hugging Face dataset repository identifier.
    """

    output_dir: Path = Path("./data")
    hf_token: str | None = None
    repo_id: str = "Vision-CAIR/TVQA-Long"


@dataclass
class ModelDownloadConfig:
    """Options controlling model snapshot downloads.

    Attributes:
        model_ids: Mapping of local model names to Hugging Face repo ids.
        revision: Optional git revision to pin when downloading models.
        local_dir: Base directory where model snapshots are stored.
        token: Optional Hugging Face token for gated model repos.
    """

    model_ids: dict[str, str] = field(
        default_factory=lambda: {
            "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "video-llava-7b": "LanguageBind/Video-LLaVA-7B-hf",
            "videollama3-7b": "DAMO-NLP-SG/VideoLLaMA3-7B",
            "internvl3-8b": "OpenGVLab/InternVL3-8B",
            "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
            "minigpt4-video-mistral": "Vision-CAIR/MiniGPT4-Video",
        }
    )
    revision: str | None = None
    local_dir: Path = Path("./models")
    token: str | None = None


def load_environment_variables() -> None:
    """Load the project .env file into the process environment.

    Raises:
        FileNotFoundError: If `.env` is missing at the repository root.
    """
    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError(
            ".env file not found. Please create one using .env.example as a template."
        )

    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from {env_path.absolute()}")


def get_hf_token() -> str | None:
    """Return a usable Hugging Face token, warning if one is not configured.

    Returns:
        Token string from `HUGGING_FACE_TOKEN`, or None when unset/placeholder.
    """
    token = os.getenv("HUGGING_FACE_TOKEN")
    if token and token != "your_hf_token_here":
        return token
    else:
        print(
            "Warning: No Hugging Face token found. Download may fail for gated repos."
        )
        return None


def download_file(
    repo_id: str, filename: str, local_dir: Path, token: str | None
) -> Path:
    """Download one dataset file from Hugging Face and store it locally.

    Args:
        repo_id: Target dataset repository on the Hugging Face Hub.
        filename: Path of the file inside the repository.
        local_dir: Directory where the file should be saved.
        token: Optional Hugging Face token for gated repositories.

    Returns:
        Path to the downloaded file on disk.

    Raises:
        Exception: Propagates any download or filesystem errors.
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


def get_friends_script_files() -> list[str]:
    """List available Friends HTML script files from the GitHub repository.

    Returns:
        Alphabetized list of script filenames.

    Raises:
        Exception: If the GitHub API request fails or payload is malformed.
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
    """Download a single Friends HTML script and write it to disk.

    Args:
        script_file: Filename of the script to retrieve.
        output_dir: Directory where the script should be stored.

    Returns:
        Path to the saved script file.

    Raises:
        Exception: If the HTTP request or file write fails.
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
    """Download every Friends script from the friends-scripts GitHub project.

    Args:
        output_dir: Base directory where scripts will be organized.

    Raises:
        Exception: If any script fetch fails.
    """
    print("Downloading Friends Scripts from edersoncorbari GitHub")

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
    """Download all split TVQA-Long archives in parallel.

    Args:
        config: Download parameters including Hugging Face repo and token.

    Raises:
        Exception: If any archive part fails to download.
    """
    print(f"Downloading Dataset from repository: {config.repo_id}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    total_files = len(VIDEO_FILES)
    current_file = 0

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


def download_model(model_name: str, repo_id: str, config: ModelDownloadConfig) -> Path:
    """Download a single model snapshot into `config.local_dir/model_name`.

    Args:
        model_name: Local directory name to store the model under.
        repo_id: Hugging Face model repository identifier.
        config: Download settings including target directory and token.

    Returns:
        Filesystem path to the downloaded model snapshot.

    Raises:
        Exception: If the snapshot download fails.
    """
    target_dir = config.local_dir / model_name
    print(f"Downloading model {model_name} ({repo_id})")
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        repo_id=repo_id,
        revision=config.revision,
        local_dir=str(target_dir),
        token=config.token,
    )

    resolved_path = Path(model_path)
    print(f"Model downloaded to: {resolved_path}")
    return resolved_path


def download_all_models(config: ModelDownloadConfig) -> None:
    """Iterate through `config.model_ids` and download each model snapshot.

    Args:
        config: Collection of model identifiers and download options.
    """
    for model_name, repo_id in config.model_ids.items():
        try:
            download_model(model_name, repo_id, config)
        except Exception as exc:
            print(f"Failed to download {model_name}: {exc}")


def download_goldfish_checkpoint(token: str | None) -> Path:
    """Fetch the MiniGPT4-Video checkpoint used by goldfish, reusing existing files.

    Args:
        token: Optional Hugging Face token for gated checkpoints.

    Returns:
        Path to the local checkpoint file.

    Raises:
        Exception: If the checkpoint cannot be downloaded or saved.
    """
    checkpoints_dir = CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    filename = "video_mistral_checkpoint_last.pth"
    local_path = checkpoints_dir / filename
    if local_path.exists():
        print(f"Goldfish checkpoint already exists: {local_path}")
        return local_path

    print("Downloading goldfish MiniGPT4-Video checkpoint...")
    downloaded = hf_hub_download(
        repo_id="Vision-CAIR/MiniGPT4-Video",
        filename=f"checkpoints/{filename}",
        token=token,
    )
    downloaded_path = Path(downloaded)

    if downloaded_path != local_path:
        local_path.write_bytes(downloaded_path.read_bytes())
    print(f"Goldfish checkpoint downloaded to: {local_path}")

    return local_path


def combine_split_archives(data_dir: Path) -> Path:
    """Concatenate the downloaded split archives into one tar.gz and remove parts.

    Args:
        data_dir: Root directory holding the `tvqa-long-videos` archive parts.

    Returns:
        Path to the combined archive.

    Raises:
        FileNotFoundError: If any expected archive part is missing.
    """
    print("Combining Split Video Archives")

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
            "Please re-run download.py first."
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
    """Extract only Friends videos from the combined archive and clean leftovers.

    Args:
        archive_path: Combined tar.gz archive containing all shows.
        output_dir: Destination root where extracted videos will be stored.

    Returns:
        Path to the directory containing extracted Friends videos.

    Raises:
        FileNotFoundError: If the combined archive is missing.
    """
    print("Extracting Friends Videos Only")

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


def download_pipeline() -> None:
    """Run the full download workflow for scripts, videos, models, and checkpoints."""
    try:
        data_dir = Path("./data")

        load_environment_variables()

        hf_token = get_hf_token()

        data_config = DownloadConfig(
            output_dir=Path("./data"),
            hf_token=hf_token,
        )

        download_edersoncorbari_friends_scripts(data_config.output_dir)

        download_tvqa_dataset(data_config)

        combined_archive = combine_split_archives(data_dir)

        extract_video_archive(combined_archive, data_dir)

        model_config = ModelDownloadConfig(token=hf_token)

        download_all_models(model_config)

        download_goldfish_checkpoint(hf_token)

    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
