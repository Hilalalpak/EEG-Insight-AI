"""
Handles unzipping and validating parquet files downloaded from Kaggle.
"""
import os
import zipfile
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def prepare_parquet(local_eeg_path, eeg_id):
    """
        Ensures the file is unzipped and is a valid parquet.
        Handles .zip archives and parquet files that are secretly zips.
        """
    zip_path = local_eeg_path + ".zip"

    if os.path.exists(zip_path):
        print("  Extracting from .zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(local_eeg_path))
        os.remove(zip_path)
    elif os.path.exists(local_eeg_path):
        with open(local_eeg_path, "rb") as f:
            first_bytes = f.read(2)

        if first_bytes == b"PK":
            print("  File is zipped, extracting...")
            temp_path = local_eeg_path + ".temp"
            with zipfile.ZipFile(local_eeg_path, "r") as zip_ref:
                with zip_ref.open(os.path.basename(local_eeg_path)) as zipped_file:
                    with open(temp_path, "wb") as out_file:
                        out_file.write(zipped_file.read())
            os.remove(local_eeg_path)
            os.rename(temp_path, local_eeg_path)
        else:
            print("Parquet file ready")
    else:
        print("Error: File not found after download")
        return False

    if not os.path.exists(local_eeg_path):
        print("  Error: Parquet file missing")
        return False

    try:
        test_df = pd.read_parquet(local_eeg_path)
        print(f"Valid parquet: {len(test_df)} rows")
        return True
    except Exception as e:
        print(f"Invalid parquet: {str(e)[:80]}")
        return False

def create_metadata_file(eeg_id, eeg_dir, train_df):
    """Filters the main train.csv to create a local metadata.csv."""
    local_metadata_path = os.path.join(eeg_dir, "metadata.csv")

    metadata_df = train_df[train_df["eeg_id"] == int(eeg_id)]
    metadata_df.to_csv(local_metadata_path, index=False)

    return local_metadata_path