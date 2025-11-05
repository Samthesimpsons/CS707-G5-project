# CS707-G5-project
Project for CS707 G5

[OUTDATED]

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

### 4. Set Up API Keys

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`:
   ```
   # Hugging Face (for datasets)
   HUGGING_FACE_TOKEN=hf_your_token_here

   # OpenAI (required for data processing pipeline)
   OPENAI_API_KEY=add_your_key_here
   ```

Get keys from:
- Hugging Face: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

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

## Usage

### Running the Pipelines

The project includes pipeline tasks that should be run in sequence:

TBD

## Development

### Code Quality Checks

Run linting, formatting, and type checks:
```bash
poe checks
```