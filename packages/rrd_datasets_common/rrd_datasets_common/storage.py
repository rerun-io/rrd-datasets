"""
Storage backend config shared by the examples.

A few environment variables pick the storage backend and configure it.

`STORAGE_BACKEND` picks `s3` (AWS S3, the default) or `hf` (a HuggingFace Bucket behind the
S3-compatible gateway).

Each backend brings its own variables:
export `S3_BUCKET` / `S3_REGION` for s3, or `HF_NAMESPACE` / `HF_BUCKET` for hf, or edit the
placeholders below.

Everything one dataset owns sits under `dataset_prefix(...)`; each example declares its layout
beneath it in its own `storage.py`. On `s3` the prefix is `s3://<bucket>/<dataset>/` — an AWS
bucket may be shared, so the dataset nests under its own segment. On `hf` it is `s3://<bucket>/`
— the bucket is created for one dataset, so the layout sits at its root.

The `s3://` shape holds on both backends — HF bucket keys are addressed the same way once the
client points at the gateway.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------------------
# S3 / HF Bucket Config
# --------------------------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "<your-bucket>")
S3_REGION = os.getenv("S3_REGION", "<your-region>")

HF_NAMESPACE = os.getenv("HF_NAMESPACE", "<your-namespace>")
HF_BUCKET = os.getenv("HF_BUCKET", "<your-hf-bucket>")

# --------------------------------------------------------------------------------------
# Backend Selection
# --------------------------------------------------------------------------------------
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "s3")
if STORAGE_BACKEND not in ("s3", "hf"):
    raise SystemExit(f"STORAGE_BACKEND must be 's3' or 'hf', got {STORAGE_BACKEND!r}.")

BUCKET = S3_BUCKET if STORAGE_BACKEND == "s3" else HF_BUCKET


def dataset_prefix(dataset: str) -> str:
    """The bucket prefix everything `dataset` owns sits under (see the module docstring)."""
    return f"s3://{BUCKET}/{dataset}/" if STORAGE_BACKEND == "s3" else f"s3://{BUCKET}/"
