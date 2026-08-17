"""
Download the sample episodes from the BitRobot/HIW-500 dataset.

HIW-500 ("Humanoid In the Wild") is an open teleoperation dataset collected on Unitree G1
humanoids with Dex1 grippers. Each episode is a directory holding one ROS2 MCAP plus small
sidecars. This script grabs four episodes (~450 MB) so there is something to poke at
locally without pulling the full dataset.

Run:  pixi run -e hiw download
"""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from hiw_500.episode_index import CACHE_PATH, HF_REPO_ID
from rrd_datasets_common.hf_repo import hf_file_index
from rrd_datasets_common.paths import dataset_data_dir

SAMPLES = [
    "Sweep-Floor/episode_2026-05-19_14-35-04/episode_0006",  # 88 MB
    "Clothes-Washing/episode_2026-05-25_16-41-06/episode_0002",  # 112 MB
    "Picking-Trash-To-Rubbish-Bin/episode_2026-04-13_09-30-35/episode_0002",  # 178 MB
    "Move-The-Pillow-To-The-Sofa-From-Floor/episode_2026-02-24_14-31-06/episode_0001",  # 76 MB, no IR
]
# The dataset structure looks like the following:
#
# BitRobot/HIW-500/
# ├── README.md
# ├── assets/
# └── <task_name>/                       # e.g. Sweep-Floor
#     └── episode_<datetime>/            # one recording session
#         └── episode_NNNN/
#             ├── episode_NNNN.mcap      # trajectory: cameras, joint state, grippers, odometry
#             ├── info.json              # task label, subtask boundaries, scene id
#             └── calibration/params/    # camera calibration — some episodes ship none
#
LOCAL_DIR = dataset_data_dir("HIW-500")


def main() -> None:
    # Which files an episode ships varies, so the repo's cached file index says what to fetch.
    files = hf_file_index(HF_REPO_ID, CACHE_PATH)
    print(f"Downloading {len(SAMPLES)} sample episodes…")
    for sample in SAMPLES:
        sample_files = sorted(path for path in files if path.startswith(f"{sample}/"))
        if not sample_files:
            raise RuntimeError(f"No files found at {HF_REPO_ID}/{sample}")
        print(f"Downloading {sample}…")
        for filename in sample_files:
            hf_hub_download(  # It skips downloading if the file already exists.
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=LOCAL_DIR,
            )
        print(f"Downloaded sample episode to {LOCAL_DIR}/{sample}")
    print("Downloaded all sample episodes.")


if __name__ == "__main__":
    main()
