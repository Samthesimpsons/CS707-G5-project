import gc
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import torch
from goldfish.inference_utils_batch import generate_batch, load_question_answer_pairs  # ty: ignore
from goldfish.model import VideoLlava  # ty: ignore
from goldfish.paths import (  # ty: ignore
    MODELS_DIR,
    QA_OUTPUT_DIR,
    RESULTS_DIR,
    VIDEOS_DIR,
    ensure_dir,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_CLASSES: dict[str, type] = {
    # "qwen-vl-2": QwenVL,
    # "qwen-vl-25": QwenVL,
    "video-llava": VideoLlava,
    # "video-llama-3": VideoLLama3,
    # "intern-vl-3": InternVL3,
}

DEFAULT_MODEL_PATHS: dict[str, str] = {
    "video-llava": str(MODELS_DIR / "video-llava-7b"),
}


@dataclass
class GoldfishConfig:
    """Configuration options for running goldfish inference batches.

    Attributes:
        qa_pairs_dir: Directory containing QA JSON files.
        video_dir: Directory containing video clips.
        results_dir: Destination directory for inference outputs.
        model_paths: Mapping from model key to filesystem path.
        max_new_tokens: Maximum tokens to generate per response.
        neighbours: List of neighbor counts for embedding-based retrieval.
        captions: Caption types to use when generating prompts.
        use_openai_embedding: Whether to use OpenAI embeddings.
        runs: Number of repeated runs per configuration.
    """
    qa_pairs_dir: Path = QA_OUTPUT_DIR
    video_dir: Path = VIDEOS_DIR
    results_dir: Path = RESULTS_DIR
    model_paths: dict[str, str] = field(
        default_factory=lambda: DEFAULT_MODEL_PATHS.copy()
    )
    max_new_tokens: int = 128
    neighbours: list[int] = field(default_factory=lambda: [1, 3, 5])
    captions: list[str] = field(default_factory=lambda: ["generic", "specific"])
    use_openai_embedding: bool = False
    runs: int = 1


def resolve_model_paths(overrides: Iterable[str] | None) -> dict[str, str]:
    """Resolve model identifiers to filesystem paths.

    Args:
        overrides: Optional iterable of model keys or `key=path` overrides.

    Returns:
        Mapping from model key to resolved path.

    Raises:
        ValueError: If an unknown model key is provided without a path.
    """
    paths: dict[str, str] = DEFAULT_MODEL_PATHS.copy()
    if overrides:
        paths.clear()
        for override in overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                paths[key.strip()] = value.strip()
            else:
                key = override.strip()
                if key not in DEFAULT_MODEL_PATHS:
                    available = ", ".join(DEFAULT_MODEL_PATHS)
                    raise ValueError(
                        f"Unknown model '{key}'. Use one of: {available} "
                        "or supply key=/path override."
                    )
                paths[key] = DEFAULT_MODEL_PATHS[key]
    return paths


def run_model_inference(
    model_name: str,
    model_path: str,
    qa_pairs: list[dict],
    config: GoldfishConfig,
) -> None:
    """Run inference for a single model across configured runs/captions/neighbours.

    Args:
        model_name: Key identifying which model class to instantiate.
        model_path: Filesystem path to the model weights.
        qa_pairs: Loaded QA records to evaluate.
        config: Goldfish configuration bundle.

    Raises:
        ValueError: If no model class is registered for the given key.
    """
    print(f"\n{'='*80}")
    print(f"Running inference for model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*80}\n")

    gc.collect()
    torch.cuda.empty_cache()
    print("Torch CUDA cache emptied")

    model_class = MODEL_CLASSES.get(model_name)
    if model_class is None:
        available_models = ", ".join(MODEL_CLASSES.keys())
        raise ValueError(
            f"No model class found for {model_name} - Available models: {available_models}"
        )

    model = model_class(model_path=model_path)
    print(f"Inference: Model loaded from {model_path}")

    for run_id in range(1, config.runs + 1):
        for caption in config.captions:
            for neighbour in config.neighbours:
                output_dir = (
                    config.results_dir
                    / model_name
                    / f"run_{run_id}"
                    / f"caption_{caption}"
                    / f"nn_{neighbour}"
                )
                ensure_dir(output_dir)
                generate_batch(
                    model,
                    qa_pairs,
                    str(config.qa_pairs_dir),
                    str(config.video_dir),
                    output_path=str(output_dir),
                    max_new_tokens=config.max_new_tokens,
                    neighbours=neighbour,
                    use_openai_embedding=config.use_openai_embedding,
                    caption_type=caption,
                )


def goldfish_pipeline(
    qa_pairs_dir: Path | str = QA_OUTPUT_DIR,
    video_dir: Path | str = VIDEOS_DIR,
    results_dir: Path | str = RESULTS_DIR,
    model: Iterable[str] | None = None,
    model_paths: dict[str, str] | None = None,
    runs: int = 1,
    neighbours: list[int] | None = None,
    captions: list[str] | None = None,
    max_new_tokens: int = 128,
    use_openai_embedding: bool = False,
) -> None:
    """Execute goldfish inference across one or more models.

    Args:
        qa_pairs_dir: Directory containing QA JSON files.
        video_dir: Directory containing video clips.
        results_dir: Destination directory for outputs.
        model: Optional iterable of model keys or key/path overrides.
        model_paths: Optional explicit mapping of model key to path.
        runs: Number of runs per model configuration.
        neighbours: Neighbor counts for embedding-based retrieval.
        captions: Caption types to include.
        max_new_tokens: Maximum tokens to generate per response.
        use_openai_embedding: Whether to use OpenAI embeddings.

    Raises:
        ValueError: If no QA pairs are found in the specified directory.
    """
    resolved_model_paths = (
        dict(model_paths)
        if model_paths is not None
        else resolve_model_paths(model)
    )
    config = GoldfishConfig(
        qa_pairs_dir=Path(qa_pairs_dir),
        video_dir=Path(video_dir),
        results_dir=Path(results_dir),
        model_paths=resolved_model_paths,
        neighbours=neighbours or [1, 3, 5],
        captions=captions or ["generic", "specific"],
        runs=runs,
        max_new_tokens=max_new_tokens,
        use_openai_embedding=use_openai_embedding,
    )

    ensure_dir(config.results_dir)

    qa_pairs = load_question_answer_pairs(config.qa_pairs_dir)
    if not qa_pairs:
        raise ValueError(f"No QA pairs found at {config.qa_pairs_dir}")

    for model_name, model_path in config.model_paths.items():
        run_model_inference(model_name, model_path, qa_pairs, config)


DEFAULT_CONFIG = GoldfishConfig()


def run_goldfish(
    config: GoldfishConfig | None = None,
    model_overrides: Iterable[str] | None = None,
) -> None:
    """Run the goldfish pipeline with a provided config or overrides.

    Args:
        config: Optional GoldfishConfig; defaults to DEFAULT_CONFIG when None.
        model_overrides: Model keys or key=path overrides to replace model_paths.
    """
    base_config = config or DEFAULT_CONFIG

    # Prevent accidental mutation of the input config by cloning when overriding models.
    effective_config = (
        replace(base_config, model_paths=resolve_model_paths(model_overrides))
        if model_overrides is not None
        else base_config
    )
    goldfish_pipeline(
        qa_pairs_dir=effective_config.qa_pairs_dir,
        video_dir=effective_config.video_dir,
        results_dir=effective_config.results_dir,
        model_paths=effective_config.model_paths,
        runs=effective_config.runs,
        neighbours=effective_config.neighbours,
        captions=effective_config.captions,
        max_new_tokens=effective_config.max_new_tokens,
        use_openai_embedding=effective_config.use_openai_embedding,
    )
