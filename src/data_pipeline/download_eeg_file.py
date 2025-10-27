"""
Kaggle API wrapper for downloading competition files.
"""
import os
import kaggle
import logging

logger = logging.getLogger(__name__)

def download(file_path: str, eeg_dir: str, competition_name: str):
    """Downloads a single file from the Kaggle competition."""
    local_file_name = os.path.basename(file_path)
    eeg_id = os.path.splitext(local_file_name)[0]
    local_eeg_path = os.path.join(eeg_dir, local_file_name)

    logger.info(f"Downloading {eeg_id} via Kaggle API...")

    try:
        kaggle.api.competition_download_file(competition_name, file_path, path=eeg_dir)
        return local_eeg_path, eeg_id
    except Exception as e:
        logger.error(f"  Download error for {eeg_id}: {e}")
        return None, None