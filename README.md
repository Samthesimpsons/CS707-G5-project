# CS707-G5-project

Pipelines for the TVQA-Long (Friends) dataset and video QA models live under `scripts/`. Use the `poe` tasks to run each stage.

## Prerequisites
- Python 3.9–3.13 (see `.python-version`); install via `uv python install 3.13` if needed.
- [uv](https://github.com/astral-sh/uv) for environment management.
- CMake and FFmpeg on your PATH (CMake for some deps, FFmpeg for video handling).
- Optional: NVIDIA GPU + CUDA for model inference.

## Environment setup
```bash
# create venv and install core deps from pyproject.toml
uv sync

# goldfish/video model stack (includes torch/cu118 wheels)
pip install -r scripts/goldfish/requirements.txt
```

Copy `.env.example` to `.env` and fill in:
- `HUGGING_FACE_TOKEN` (needed for gated downloads)
- `OPENAI_API_KEY` (required for OpenAI/LiteLLM paths in `scripts/process.py` and optional in goldfish for embeddings)

## Pipelines (via poe tasks - inorder)
- `poe run-download` — Downloads Friends scripts, TVQA-Long split archives (~50GB), combines/extracts Friends videos, fetches model checkpoints to `./models` and `./goldfish/checkpoints`.
- `poe run-preprocess` — Converts downloaded HTML scripts to JSON and prepares annotated tuples in `./data/annotated_tuples` (expects manual annotation of clip timings after this step).
- `poe run-process` — Runs event extraction + QA generation into `./data/output_with_events` and `./data/qa_output`. Uses OpenAI by default; set `OPENAI_API_KEY`.
- `poe run-vanilla-inference` — Runs baseline model inference over QA pairs using videos in `./data/video/` and models in `./models/`.
- `poe run-goldfish-inference` — Runs the goldfish pipeline with nearest-neighbor captions (`generic`/`specific`) over QA pairs and Friends videos.
- `poe run-eval` — Aggregates results under `results_vanilla`/`results_goldfish` and writes summaries to `./overall_summary/`.

## Notes
- Goldfish uses precomputed embeddings/checkpoints under `scripts/goldfish/new_workspace` and downloads a MiniGPT4-Video checkpoint if missing.
- Large downloads: TVQA-Long video parts (~50GB) and multiple model weights; ensure disk space.
