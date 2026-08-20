"""
Modal jobs for HIW-500 mcap -> rrd conversion.

`convert_episodes` defines the job itself; the dataset-independent helpers live in
`rrd_datasets_common.modal_jobs`. The list of episodes to convert comes from `hiw_500.episode_index`.
"""

from __future__ import annotations
