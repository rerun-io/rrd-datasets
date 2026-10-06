"""
Download sample file groups from the allenai/MolmoAct2-BimanualYAM-Dataset.

MolmoAct2-BimanualYAM is a LeRobot v3 dataset of bimanual tabletop manipulation on two I2RT YAM
arms: 32,246 teleoperated episodes (about 704 hours) with three cameras stored as AV1 video. This
script grabs `meta/` and three file groups (~1.1 GB, 19 episodes) so there is something to poke
at locally without pulling the full ~2.4 TB.

Run:  pixi run -e molmo download
"""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from molmoact2.episode_index import HF_REPO_ID, HF_REVISION, LOCAL_DIR, discover_groups

# Each sample is a file group, named by its first episode.
SAMPLES = [
    800,  # 0.66 GB, 10 episodes of "spell out CVPR"
    1825,  # 0.37 GB, 8 episodes of "Pack container"; episode 1832 has no motion
    31673,  # 0.04 GB, a group of a single "Fold the cloth" episode
]
# The dataset structure looks like the following:
#
# allenai/MolmoAct2-BimanualYAM-Dataset/
# ├── README.md
# ├── meta/
# │   ├── info.json                         # features, fps, file path templates
# │   ├── stats.json                        # dataset-wide statistics
# │   ├── tasks.parquet                     # the 34 task strings
# │   ├── tasks_annotated.parquet           # one annotated instruction per episode
# │   └── episodes/chunk-000/file-000.parquet
# │                                         # per episode: length, task, file locations, statistics
# ├── data/chunk-NNN/file-NNN.parquet       # state, action, timestamps; about 10 episodes per file
# └── videos/observation.images.<camera>/chunk-NNN/file-NNN.mp4
#                                           # top, left, right; about 12 to 19 episodes per file
#


def main() -> None:
    groups = {group.episodes[0]: group for group in discover_groups()}
    print(f"Downloading {len(SAMPLES)} sample file groups…")
    for first_episode in SAMPLES:
        group = groups.get(first_episode)
        if group is None:
            raise RuntimeError(f"No file group starts at episode {first_episode} in {HF_REPO_ID}")
        print(f"Downloading {group.group_id} ({len(group.files)} files)…")
        for filename in group.files:
            hf_hub_download(  # It skips downloading if the file already exists.
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=HF_REVISION,
                filename=filename,
                local_dir=LOCAL_DIR,
            )
    print(f"Downloaded all samples to {LOCAL_DIR}.")


if __name__ == "__main__":
    main()
