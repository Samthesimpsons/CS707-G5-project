# CS707-G5-project

Episodic Memory Question-Answering System for Video Narratives

This project implements an automated pipeline for generating question-answering datasets from video subtitle data, focusing on episodic memory and temporal reasoning.

## Prerequisites

### 1. Install uv (Package Manager)

This project uses [uv](https://github.com/astral-sh/uv) as the package manager. Install it yourself following the official instructions.

After installing uv, run the following command to install all project dependencies from `pyproject.toml`:
```bash
uv sync
```

This will create a `.venv` directory with all dependencies installed.

### 2. Install CMake

CMake is required for HuggingFace `datasets` libary dependencies.

**Windows:**
- Download the standalone binary **Windows x64 Installer** (.msi file) from [cmake.org/download](https://cmake.org/download/)
- Run the `.msi` file and follow the installation instructions
- Make sure to check the box to add CMake to system PATH during installation

**macOS:**
```bash
brew install cmake
```

**Linux:**
```bash
sudo apt-get install cmake  # Debian/Ubuntu
sudo yum install cmake      # RHEL/CentOS
```

### 3. Python Version Setup

This project requires Python 3.9-3.13 for PyArrow compatibility.

**If you have Python 3.13 or lower, you can skip this step.**

**If you have Python 3.14+, install Python 3.13 using uv:**
```bash
uv python install 3.13
```

The `.python-version` file in the project root specifies Python 3.13, so `uv` will automatically use it.

### 4. Set Up Environment Variables

1. Copy `.env.example` to `.env`
2. Modify accordingly

### 5. Install FFmpeg (for video processing)

**Windows:**
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Extract and add to system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg  # Debian/Ubuntu
sudo yum install ffmpeg      # RHEL/CentOS
```

## Pipeline Overview

The project consists of two main stages:

### Stage 1: Batch Job Submission (`add_events.py`)
- **Input**: Annotated subtitle JSON files with scene information
- **Process**: Creates batch requests for event extraction and submits to OpenAI's Batch API
- **Features**:
  - Uses OpenAI Batch API for cost-effective processing (50% discount)
  - Generates JSONL batch request files with structured output schema
  - Returns batch job ID and tracking URL for monitoring
  - Saves batch metadata for later result processing
- **Output**: Batch job submission with tracking information (batch completes within 24 hours)
- **Note**: This script only submits batch jobs; downloading results and QA generation happen in Stage 2

### Stage 2: Batch Results Processing & QA Generation (`generate_qa.py`)
- **Input**: Batch job ID and metadata file from Stage 1
- **Process**:
  1. Downloads completed batch results from OpenAI and merges events back into episode JSON files
  2. Rule-based generation of multiple question types from extracted events
- **Features**:
  - **Batch Processing**: Checks batch completion status before downloading, parses batch output JSONL, maps results to original scenes, and creates enhanced JSON files with event annotations
  - **QA Generation**: Async parallel processing with configurable concurrency, cross-episode temporal reasoning, and comprehensive statistics reporting
- **Question Types**:
  - Single target recall (E+S→L, L+E→S, L+S→E): ~60%
  - Boolean verification: ~20%
  - Temporal ordering (multi-episode span): ~15%
  - Latest event retrieval: ~5%
  - Location-based temporal ordering: 2 per episode
- **Output**: Enhanced episode JSON files with events and QA dataset JSON files with questions and answers

## Usage

### Running the Pipeline

#### Stage 1: Submit Batch Job for Event Extraction

```bash
poe run-events-generation
```

This will:
- Create batch requests for event extraction from subtitle files
- Submit the batch to OpenAI's Batch API
- Print batch job ID, tracking URL, and metadata file path
- Display the exact command to run for Stage 2

**Output**: The script will print the complete command with arguments for Stage 2.

#### Stage 2: Download Results and Generate QA Datasets

After the batch completes (check the tracking URL from Stage 1), run with just the batch ID:

```bash
poe run-qa-generation <batch_id>
```

**Example**:
```bash
poe run-qa-generation batch_abc123xyz
```

The metadata file will be automatically located based on the batch ID.

This will:
1. Download and process batch results from OpenAI
2. Merge extracted events into episode JSON files
3. Generate QA datasets with multiple question types
4. Save outputs to `data/qa_output/`

**Advanced Usage**:
- Explicitly specify metadata file: `poe run-qa-generation <batch_id> <metadata_file_path>`
- Skip batch downloading (QA generation only): `poe run-qa-generation`

### Available Poe Tasks

The project includes the following poe tasks defined in `pyproject.toml`:

- `poe run-events-generation` - Stage 1: Submit OpenAI Batch API jobs for event extraction from subtitles
- `poe run-qa-generation [batch_id]` - Stage 2: Download batch results and generate QA datasets (metadata auto-detected)
- `poe run-check` - Run linting, formatting, and type checking

## Development

### Code Quality Checks

Run linting, formatting, and type checks:
```bash
poe run-check
```

This command runs:
1. `uvx ruff check src --fix` - Linting with auto-fix
2. `uvx black src` - Code formatting
3. `uvx ty check src` - Type checking