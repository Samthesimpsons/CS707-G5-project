from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


VISION_MODELS = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    # "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    # "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "llava-video-llama": "weizhiwang/LLaVA-Video-Llama-3.1-8B",
}

# Set cache directory to models/ folder
cache_dir = Path(__file__).parent.parent.parent / "models"


def download_model(
    model_name: str,
    model_id: Optional[str] = None,
) -> None:
    """Download a single model from HuggingFace Hub.

    Args:
        model_name: Short name for the model (e.g., "qwen2.5-vl-7b").
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-VL-7B-Instruct").
                  If None, uses VISION_MODELS mapping.

    Returns:
        None. Prints download progress and completion message.
    """
    if model_id is None:
        if model_name not in VISION_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {', '.join(VISION_MODELS.keys())}"
            )
        model_id = VISION_MODELS[model_name]

    print(f"Downloading {model_name} ({model_id})...")

    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        resume_download=True,  # Resume if download was interrupted
        local_files_only=False,  # Download from HuggingFace
    )

    print(f"Model {model_name} downloaded successfully to: {local_path}")
    return None


def download_all_models() -> None:
    """Download all vision language models.

    Returns:
        None. Prints download progress and handles errors for each model.
    """
    print("=" * 80)
    print("Vision Language Model Download Pipeline")
    print(f"Downloading {len(VISION_MODELS)} vision language models...")
    print("=" * 80)

    for model_name, model_id in VISION_MODELS.items():
        try:
            download_model(model_name, model_id)
        except Exception as e:
            print(f"Failed to download {model_name}: {e}\n")

    return None


def download_model_pipeline() -> None:
    """Download pipeline to download all vision language models.

    This is the main entry point for the model download pipeline.
    Downloads all models defined in VISION_MODELS dictionary.
    """
    try:
        download_all_models()
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
