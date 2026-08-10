"""
Modal building blocks shared by the examples.

`aws`, `hf_bucket`, and `image` hold the dataset-independent helpers; `store` dispatches on
`STORAGE_BACKEND`. Each example's own `modal_jobs/convert_episodes.py` defines its job on top.
"""

from __future__ import annotations
