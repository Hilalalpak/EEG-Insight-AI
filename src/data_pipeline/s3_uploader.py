import os
import boto3
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

def create_s3_client(endpoint_url):
    """Creates a boto3 client for a local S3 endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),)
def ensure_bucket_exists(s3_client, bucket_name):
    """Check if a bucket exists, create it if not."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except:
        s3_client.create_bucket(Bucket=bucket_name)
def upload_eeg_files(s3_client, bucket_name, eeg_id, local_eeg_path, local_metadata_path):
    """Uploads the eeg parquet and metadata csv to S3."""
    # The file path in S3 will be e.g., <bucket>/1001487592/1001487592.parquet
    local_file_name = os.path.basename(local_eeg_path)

    s3_client.upload_file(local_eeg_path, bucket_name, f"{eeg_id}/{local_file_name}")
    s3_client.upload_file(local_metadata_path, bucket_name, f"{eeg_id}/metadata.csv")

    print("Uploaded to S3")