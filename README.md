# CS707-G5-project

This project implements a video question-answering system using the TVQA-Long dataset with Friends TV show episodes. It compares baseline (vanilla) and Goldfish-based inference approaches for long-form video understanding.

## Prerequisites
- Python 3.9-3.13
- [uv](https://github.com/astral-sh/uv) for environment management
- CMake and FFmpeg on your PATH (CMake for HuggingFace deps, FFmpeg for video handling)
- Recommended: NVIDIA GPU + CUDA for efficient model inference

## Environment setup

### Using UV

This project uses `uv`, a modern and fast Python package manager written in Rust, instead of traditional `requirements.txt` and pip. `uv` provides faster dependency resolution, better reproducibility, and integrated virtual environment management.

After cloning or unzipping the project:

1. Run `uv sync` to create the `.venv` directory and download all dependencies based on `pyproject.toml`:
   ```bash
   uv sync
   ```

2. The virtual environment will be created automatically. In subsequent terminal sessions, activate it manually if it doesn't activate automatically:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate  # Windows
   ```

### Setting up API keys

- **HuggingFace token** (required): Needed to download necessary models and the TVQA-Long dataset from HuggingFace Hub
- **OpenAI API key** (optional): Only required if you use OpenAI models for QA generation or as embeddings in Goldfish inference. If using DeepSeek or other alternatives, this can be omitted.

To configure API keys:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and replace the placeholder values with your actual keys:
   ```
   HUGGING_FACE_TOKEN=your_token_here
   OPENAI_API_KEY=your_key_here  # Optional
   ```

3. No need to manually source the file - the scripts automatically load environment variables using `python-dotenv`.

## Running

This project does not provide a traditional CLI with a `main()` entry point. There are no advanced workflow orchestration tools like Kedro or Temporal. Instead, we use **poe** (`poethepoet`), which is included as a dev dependency and integrates seamlessly with `UV`.

To see available commands and help:
```bash
poe
```

### Pipelines (in order)

The project consists of six main pipeline tasks that should generally be run in sequence:

1. **run-download**: Download Friends scripts from GitHub, TVQA-Long video parts (~50GB), combine/extract Friends videos, and pull required model checkpoints from HuggingFace.
   ```bash
   poe run-download
   ```

2. **run-preprocess**: Convert HTML scripts to JSON format and produce annotated tuples with scene metadata, dialogue, and timing information (requires downloaded scripts/videos).
   ```bash
   poe run-preprocess
   ```

3. **run-process**: Extract atomic events from scenes using LLMs and generate QA pairs from annotated tuples.
   ```bash
   poe run-process
   ```

4. **run-vanilla-inference**: Run baseline video-QA inference over generated QA pairs and videos, saving results to `results_vanilla`.
   ```bash
   poe run-vanilla-inference
   ```

5. **run-goldfish-inference**: Run Goldfish nearest-neighbor captioned inference over QA pairs and videos, saving results to `results_goldfish`.
   ```bash
   poe run-goldfish-inference
   ```

6. **run-evaluation**: Aggregate results from vanilla or goldfish inference into summary reports for analysis.
   ```bash
   poe run-evaluation
   ```

Additionally, there is a code quality check task:
- **run-check**: Run linting (ruff), formatting (black), and static type checking (ty).
  ```bash
  poe run-check
  ```

## Sample Data

The repository includes sample data from various stages of the pipeline in the `data/` directory to help you understand the expected outputs at each step:

- **`edersoncorbari_subtitles/`**: Raw HTML subtitle files downloaded after `run-download`. Contains Season 1, Episode 1 only as a sample.

- **`edersoncorbari_subtitles_json/`**: JSON-formatted subtitles after `run-preprocess` converts HTML to structured JSON. Contains Season 1, Episode 1 only as a sample.

- **`annotated_tuples/`**: Scene metadata with manually annotated timing information produced by `run-preprocess`. Contains the Season 1.

- **`output_with_events/`**: Annotated tuples with extracted atomic events added by the `run-process` pipeline. Contains the Season 1.

- **`qa_output/`**: Generated question-answer pairs from `run-process`. Includes `finalised_qa.json`, which is the combined QA dataset used in our studies.

- **`videos/`**: Sample video files. Contains Season 1, Episode 1 only as a sample.

## Notes
- **Storage requirements**: TVQA-Long video parts are approximately 50GB, plus additional space needed for model weights and intermediate outputs. Ensure sufficient disk space before running the download pipeline.
- The `scripts/goldfish` directory contains adapted code and is excluded from linting/formatting checks.
- Video processing and inference can be computationally intensive. A GPU is highly recommended for reasonable execution times.

## Acknowledgements
- [Goldfish](https://github.com/bigai-nlco/goldfish) - For the video understanding framework and methodology
- [TVQA-Long](https://github.com/jayleicn/TVQAplus) - For the dataset and evaluation framework
- [edersoncorbari/friends-scripts](https://github.com/edersoncorbari/friends-scripts) - For the Friends episode scripts used in preprocessing