"""
Modal jobs for ABC-130k mcap -> rrd conversion.

`convert_episodes` defines the job itself; the dataset-independent helpers live in
`rrd_datasets_common.modal_jobs`. The list of episodes to convert comes from `abc_130k.episode_index`.
"""

from __future__ import annotations
