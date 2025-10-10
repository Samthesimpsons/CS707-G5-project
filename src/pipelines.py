"""
Pipeline orchestration for CS707 G5 Project.
"""

from data.preprocessing import preprocessing_pipeline  # type: ignore[import-not-found]
from data.processing import processing_pipeline  # type: ignore[import-not-found]


def start_data_pipeline() -> None:
    """
    Download, extract, preprocess and submits a batch job for Friends dataset.
    """
    preprocessing_pipeline()


def resume_data_pipeline() -> None:
    """
    Resume data processing after batch job completes.
    """
    processing_pipeline()
