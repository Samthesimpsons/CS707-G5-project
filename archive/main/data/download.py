import os
import requests  # type: ignore[import-untyped]
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
from dotenv import load_dotenv  # type: ignore[import-not-found]


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

SUBTITLE_FILE = "tvqa_subtitles.zip"

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


def download_data_pipeline() -> None:
    """Download pipeline to orchestrate dataset download.

    This function:
    1. Loads environment variables from .env file
    2. Gets Hugging Face token
    3. Downloads videos and raw subtitles from TVQA-Long (Hugging Face)
    4. Downloads Friends scripts with speaker labels from GitHub
    """
    try:
        load_environment_variables()

        hf_token = get_hf_token()

        config = DownloadConfig(
            output_dir=Path("./data"),
            hf_token=hf_token,
        )

        download_edersoncorbari_friends_scripts(config.output_dir)

        download_tvqa_dataset(config)

    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
