"""
Download a single sample episode from the XDOF/ABC-130k dataset.

ABC-130k is a large open-source bimanual robot-teleoperation dataset collected on
two-arm YAM stations. Each episode is distributed as a single MCAP file. This grabs
one episode (~450 MB) from the "fold and stack the t-shirts" task so there is
something to poke at locally without pulling the full (>1 TB) dataset.

Run:  pixi run -e abc download
"""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from abc_130k.episode_index import HF_REPO_ID
from rrd_datasets_common.paths import dataset_data_dir

SAMPLE = "data/train/fold_and_stack_the_t_shirts/episode_001005fe-c6ed-4e3c-b6ce-6beb4e8ce0cf"
# The dataset namespaces its files under `data/`, so the sample lands at
# data/ABC-130k/data/train/... under the shared data root (repo-relative path
# preserved beneath it).
LOCAL_DIR = dataset_data_dir("ABC-130k")

# Fetch the known files directly: a pattern-based snapshot_download would first
# list all ~305k repo files, which takes minutes with no output.
FILES = ("episode.mcap", "annotation.mcap")


def main() -> None:
    for filename in FILES:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=f"{SAMPLE}/{filename}",
            local_dir=LOCAL_DIR,
        )
    print(f"Downloaded sample episode to {LOCAL_DIR}/{SAMPLE}")


if __name__ == "__main__":
    main()
