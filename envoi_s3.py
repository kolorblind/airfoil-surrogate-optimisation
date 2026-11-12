"""
Upload the results (CSV and figures) to an S3-compatible bucket,
for example Scaleway Object Storage.

Configured only through environment variables, so that no secret is stored
in the image:
    S3_BUCKET              bucket name (if unset, nothing is uploaded)
    S3_ENDPOINT            default: https://s3.fr-par.scw.cloud
    S3_REGION              default: fr-par
    S3_PREFIX              folder in the bucket, default: run-<date>-<time>
    AWS_ACCESS_KEY_ID      access key
    AWS_SECRET_ACCESS_KEY  secret key
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

bucket = os.environ.get("S3_BUCKET")
if not bucket:
    print("S3_BUCKET not set: results kept locally only.")
    sys.exit(0)

endpoint = os.environ.get("S3_ENDPOINT", "https://s3.fr-par.scw.cloud")
region = os.environ.get("S3_REGION", "fr-par")
prefix = os.environ.get(
    "S3_PREFIX", datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S"))

s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region)

fichiers = sorted(p for p in Path(".").iterdir()
                  if p.is_file() and p.suffix in {".csv", ".png"})
if not fichiers:
    print("No results to upload.")
    sys.exit(1)

for f in fichiers:
    cle = f"{prefix}/{f.name}"
    s3.upload_file(str(f), bucket, cle)
    print(f"uploaded s3://{bucket}/{cle}")
print(f"{len(fichiers)} files uploaded to {endpoint}")
