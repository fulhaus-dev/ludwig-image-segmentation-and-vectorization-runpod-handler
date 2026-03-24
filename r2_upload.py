"""
Cloudflare R2 upload for product cutout images.
Uses boto3 S3-compatible API.
"""

import os
import boto3

# Initialized once at container startup
_s3_client = None
_bucket = None
_folder = None
_public_url = None


def init_r2():
    """Initialize R2 client. Called once at container startup."""
    global _s3_client, _bucket, _public_url, _folder

    _s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto",
    )
    _bucket = os.environ["R2_BUCKET"]
    _folder = os.environ["R2_BUCKET_FOLDER"]
    _public_url = os.environ["R2_PUBLIC_URL"].rstrip("/")


def upload_cutout(sku: str, png_bytes: bytes) -> str:
    """
    Upload PNG cutout to R2.
    Returns public URL.
    """
    key = f"{_folder}/{sku}.png"

    _s3_client.put_object(
        Bucket=_bucket,
        Key=key,
        Body=png_bytes,
        ContentType="image/png",
    )

    return f"{_public_url}/{key}"
