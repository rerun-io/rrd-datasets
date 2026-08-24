"""
Download sample task files from the LIBERO benchmark datasets.

LIBERO is a lifelong-learning manipulation benchmark simulated in robosuite/MuJoCo: 130
tabletop tasks across five suites, with one HDF5 file per task holding ~50 teleoperated
demos. This script grabs one task per suite (~3.4 GB) so every suite has something to poke
at locally without pulling the full ~100 GB dataset.

Run:  pixi run -e libero download
"""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from libero.episodes import HF_REPO_ID, HF_REVISION, LOCAL_DIR

SAMPLES = [
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",  # 1.3 GB
    "libero_90/KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5",  # 0.5 GB, subtask of the libero_10 sample
    "libero_goal/turn_on_the_stove_demo.hdf5",  # 0.5 GB, same behavior again in a third suite
    "libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5",  # 0.7 GB
    "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",  # 0.5 GB
]
# The dataset structure looks like the following:
#
# yifengzhu-hf/LIBERO-datasets/
# ├── README.md
# └── <suite>/                  # libero_10, libero_90, libero_goal, libero_object, libero_spatial
#     └── <task>_demo.hdf5      # one file per task, ~50 demos each: actions, rewards,
#                               # joint/gripper/end-effector states, two 128×128 RGB cameras
#


def main() -> None:
    print(f"Downloading {len(SAMPLES)} sample task files…")
    for sample in SAMPLES:
        print(f"Downloading {sample}…")
        hf_hub_download(  # It skips downloading if the file already exists.
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            filename=sample,
            local_dir=LOCAL_DIR,
        )
    print(f"Downloaded all samples to {LOCAL_DIR}.")


if __name__ == "__main__":
    main()
