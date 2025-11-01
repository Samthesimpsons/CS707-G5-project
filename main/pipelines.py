"""Pipeline orchestration for CS707 G5 Project.

This module provides four main pipeline tasks:

1. run_download_model_pipeline: Downloads required vision language models from Hugging Face.
   This step is optional if models are already downloaded locally.

2. run_download_data_pipeline: Downloads the Friends TV show dataset and extracts the data.
   This step is optional if processed data zip file is provided.

3. run_process_data_pipeline: Preprocesses and processes the data, including submitting
   batch jobs for OpenAI API processing. Must wait for batch job completion before proceeding.

4. resume_run_process_data_pipeline: Resumes data processing after batch job completes,
   extracting video clips and finalizing the dataset.
"""

from data.download import download_data_pipeline  # type: ignore[import-not-found]
from data.extract import extract_data_pipeline  # type: ignore[import-not-found]
from data.preprocessing import preprocess_data_pipeline  # type: ignore[import-not-found]
from data.processing import process_data_pipeline, resume_process_data_pipeline  # type: ignore[import-not-found]
from models.download import download_model_pipeline  # type: ignore[import-not-found]


def run_download_model_pipeline() -> None:
    download_model_pipeline()


def run_download_data_pipeline() -> None:
    download_data_pipeline()
    extract_data_pipeline()


def run_process_data_pipeline() -> None:
    preprocess_data_pipeline()
    process_data_pipeline()


def resume_run_process_data_pipeline() -> None:
    resume_process_data_pipeline()
