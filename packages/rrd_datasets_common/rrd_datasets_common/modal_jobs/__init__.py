"""
Modal building blocks shared by the examples.

`aws` and `image` hold the dataset-independent helpers, and `store` holds the worker side of each
storage backend. Bucket access shared with local tasks is in `rrd_datasets_common.bucket`.
Each example's own `modal_jobs/convert_episodes.py` defines its job on top.
"""

from __future__ import annotations
