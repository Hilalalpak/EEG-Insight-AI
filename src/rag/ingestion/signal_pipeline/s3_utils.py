import os
import boto3
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import logging

load_dotenv()

class S3Utils:
    def __init__(self, logger: logging.Logger, bucket_name: str, endpoint_url: str):

        self.logger = logger
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url

        self.s3_client = self._create_s3_client()
        if not self.s3_client:
            self.logger.critical("S3 client init failed")
            raise ValueError("S3 client failed")
        if not self.bucket_name:
            self.logger.critical("BUCKET_NAME not set")
            raise ValueError("BUCKET_NAME missing")

    def _create_s3_client(self):
        """Establishes the S3 client connection."""
        try:
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"))
            self.logger.info(f"S3 client ready: {self.endpoint_url}")
            return client
        except Exception as e:
            self.logger.error(f"boto3 client error: {e}", exc_info=True)
            return None

    def list_eeg_recordings(self) -> list:
        """List all top-level 'folders' in the S3 bucket"""
        self.logger.info(f"Listing recordings from {self.bucket_name}...")
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Delimiter="/")
            recordings = response.get("CommonPrefixes", [])
            self.logger.info(f"Found {len(recordings)} EEG IDs")
            return recordings
        except Exception as e:
            self.logger.error(f"S3 list error: {e}", exc_info=True)
            return []

    def load_metadata(self, eeg_id: str) -> pd.DataFrame | None:
        metadata_key = f"{eeg_id}/metadata.csv"
        try:
            meta_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=metadata_key)
            meta_bytes = meta_obj["Body"].read()
            metadata_df = pd.read_csv(BytesIO(meta_bytes))

            filtered_df = metadata_df[metadata_df["eeg_id"] == int(eeg_id)].copy()
            if filtered_df.empty:
                self.logger.warning(f"No metadata for {eeg_id}")
            return filtered_df
        except Exception as e:
            self.logger.error(f"Metadata load error for {eeg_id}: {e}")
            return None

    def load_eeg_data(self, eeg_id: str) -> pd.DataFrame | None:
        parquet_key = f"{eeg_id}/{eeg_id}.parquet"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=parquet_key)
            file_bytes = response["Body"].read()
            return pd.read_parquet(BytesIO(file_bytes))
        except Exception as e:
            self.logger.error(f"EEG load error for {eeg_id}: {e}")
            return None