"""
Download the sample episodes from the XDOF/ABC-130k dataset.

ABC-130k is a large open-source bimanual robot-teleoperation dataset collected on
two-arm YAM stations. Each episode is distributed as a single MCAP file. This script grabs
five episodes (~900 MB) from the train split so there is
something to poke at locally without pulling the full (>20 TB) dataset.

Run:  pixi run -e abc download
"""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from abc_130k.episode_index import HF_REPO_ID
from rrd_datasets_common.paths import dataset_data_dir

SAMPLES = [  # (filepath, has_annotation)
    ("data/train/remove_the_shorts_from_the_hanger/episode_0132ab84-8dd2-4fb7-ade9-12b68989720d", False),  # 13 MB
    ("data/train/fold_and_stack_the_t_shirts/episode_001005fe-c6ed-4e3c-b6ce-6beb4e8ce0cf", True),  # 121 MB
    ("data/train/arrange_the_flowers_into_the_vase/episode_7a27e807-dec2-49cb-a184-743944cd7cd0", False),  # 377 MB
    ("data/train/clip_the_underwear_to_the_hanger/episode_0893975b-d8a5-4e92-9291-1f37c416b25a", True),  # 204 MB
    ("data/train/remove_the_keys_from_the_keyring/episode_a8a5e956-840e-45e6-a949-02b211218877", False),  # 163 MB
]
# The dataset structure looks like the following:
# (from https://huggingface.co/datasets/XDOF/ABC-130k#dataset-structure)
#
# XDOF/ABC-130k/
# ├── README.md
# ├── data/
# │   ├── train/                        # training split
# │   │   └── <task_name>/
# │   │       ├── episode_XXXX/          # UUID-based episode directory
# │   │       │   ├── episode.mcap       # trajectory: joint state, gripper, video, calibration, task name
# │   │       │   └── annotation.mcap    # subtask labels — annotated episodes only
# │   │       └── ...
# │   └── val/                           # validation split
# │       └── <task_name>/
# │           └── ...
# └── meta/
#     ├── train_report.txt               # note: actual file names are different from their description
#     └── val_report.txt                 #      (includes  task list, trajectory counts, hours)
#
LOCAL_DIR = dataset_data_dir("ABC-130k")


# Fetch the known files
FILES = ("episode.mcap", "annotation.mcap")


def main() -> None:
    print(f"Downloading {len(SAMPLES)} sample episodes…")
    for sample, has_annotation in SAMPLES:
        print(f"Downloading {sample}…")
        for filename in FILES:
            if filename == "annotation.mcap" and not has_annotation:
                continue
            hf_hub_download(  # It skips downloading if the file already exists.
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=f"{sample}/{filename}",
                local_dir=LOCAL_DIR,
            )
        print(f"Downloaded sample episode to {LOCAL_DIR}/{sample}")
    print("Downloaded all sample episodes.")


if __name__ == "__main__":
    main()
