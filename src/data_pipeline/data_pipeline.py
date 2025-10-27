"""
One-off script to download EEG data from Kaggle, process it and upload it to the S3 bucket.
"""
import os
import shutil
import pandas as pd
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_pipeline.download_eeg_file import download
from src.data_pipeline.file_processor import prepare_parquet, create_metadata_file
from src.data_pipeline.s3_uploader import create_s3_client, ensure_bucket_exists, upload_eeg_files
from infrastructure.conf.config_loader import ConfigLoader
CONFIG_PATHS = {"base": "infrastructure/conf/base.yml"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        config_loader = ConfigLoader(CONFIG_PATHS)
        db_config = config_loader.get_db_config()
        misc_config = config_loader.get_misc_config()

        bucket_name = db_config.get_s3_bucket_name()
        endpoint_url = db_config.get_s3_local_endpoint()

        competition_name = misc_config.get_kaggle_competition_name()

    except RuntimeError as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    temp_dir = "./temp_kaggle_data"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        s3_client = create_s3_client(endpoint_url)
        ensure_bucket_exists(s3_client, bucket_name)

        train_df = pd.read_csv("data/eeg/train.csv")

        target_files = [
            "train_eegs/1001487592.parquet",
            "train_eegs/1001717358.parquet",
            "train_eegs/1002136740.parquet",
            "train_eegs/1002142157.parquet",
            "train_eegs/1002197945.parquet",
            "train_eegs/1002379034.parquet",
            "train_eegs/1002576868.parquet",
            "train_eegs/1003330515.parquet",
            "train_eegs/1002858110.parquet",
            "train_eegs/1003011202.parquet"]

        for file_path in target_files:
            eeg_id_str = os.path.splitext(os.path.basename(file_path))[0]
            eeg_dir = os.path.join(temp_dir, eeg_id_str)
            os.makedirs(eeg_dir, exist_ok=True)
            logger.info(f"\nProcessing {eeg_id_str}...")

            # Download file
            local_eeg_path, eeg_id = download(file_path, eeg_dir, competition_name)
            if not local_eeg_path:
                logger.warning(f"Download failed for {eeg_id_str}, skipping.")
                continue

            # Extract and validate
            if not prepare_parquet(local_eeg_path, eeg_id):
                logger.warning(f"Parquet validation failed for {eeg_id}, skipping.")
                continue

            # Create metadata
            local_metadata_path = create_metadata_file(eeg_id, eeg_dir, train_df)

            # Upload to S3
            logger.info(f"  Uploading {eeg_id} to S3 bucket {bucket_name}...")
            upload_eeg_files(s3_client, bucket_name, eeg_id, local_eeg_path, local_metadata_path)

    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"\nTemp dir cleaned: {temp_dir}")
    logger.info("Data pipeline complete")

if __name__ == "__main__":
    main()