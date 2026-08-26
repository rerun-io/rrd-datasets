"""Shared HDF5 fixture: a miniature LIBERO task file, synthesized so no binary is committed."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

NUM_STEPS = 3
HEIGHT, WIDTH = 4, 2
SIM_T0, SIM_DT = 0.25, 0.05  # seconds; the timebase every surveyed demo follows

# LIBERO stands the arm on a table; the urdf layer reads this pose out of the MuJoCo XML.
BASE_POS = [-0.66, 0.0, 0.912]
MODEL_FILE = '<mujoco model="fixture"><worldbody><body name="robot0_base" pos="-0.66 0 0.912"/></worldbody></mujoco>'

# A plausible Panda pose, inside every `fer` joint limit so forward kinematics stays meaningful.
ARM_POSE = np.array([0.0, -0.2, 0.0, -2.3, 0.0, 2.1, 0.8])

# Shaped like the real `env_args`: nested objects, a list, and a JSON `null` (a null-typed struct field).
ENV_ARGS = {
    "type": 1,
    "env_name": "Libero_Test_Env",
    "env_kwargs": {"robots": ["Panda"], "controller_configs": {"type": "OSC_POSE", "kp": 150, "position_limits": None}},
}


def _write_demo(data: h5py.Group, demo: str, rng: np.random.Generator) -> None:
    group = data.create_group(demo)
    obs = group.create_group("obs")
    obs["agentview_rgb"] = rng.integers(0, 255, size=(NUM_STEPS, HEIGHT, WIDTH, 3), dtype=np.uint8)
    obs["eye_in_hand_rgb"] = rng.integers(0, 255, size=(NUM_STEPS, HEIGHT, WIDTH, 3), dtype=np.uint8)
    obs["ee_pos"] = rng.random((NUM_STEPS, 3))
    obs["ee_ori"] = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])  # last magnitude > π
    obs["joint_states"] = ARM_POSE + np.linspace(0.0, 0.1, NUM_STEPS)[:, None]
    # robosuite signs the two fingers against each other.
    opening = np.linspace(0.010, 0.039, NUM_STEPS)
    obs["gripper_states"] = np.stack([opening, -opening], axis=1)
    obs["ee_states"] = rng.random((NUM_STEPS, 6))
    group["actions"] = rng.random((NUM_STEPS, 7))
    group["rewards"] = np.array([0, 0, 1], dtype=np.uint8)
    group["dones"] = np.array([0, 0, 1], dtype=np.uint8)
    states = rng.random((NUM_STEPS, 5))
    states[:, 0] = SIM_T0 + SIM_DT * np.arange(NUM_STEPS)  # column 0 is MuJoCo's clock, as in the real files
    group["states"] = states
    group["robot_states"] = rng.random((NUM_STEPS, 9))
    group.attrs["model_file"] = MODEL_FILE
    group.attrs["init_state"] = group["states"][0]
    group.attrs["num_samples"] = NUM_STEPS


def write_fixture(path: Path) -> None:
    """A four-demo task file, with the demo keys that catch lexicographic ordering (`demo_10`)."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["bddl_file_name"] = "bddl_files/kitchen/turn_on_the_stove.bddl"
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        data.attrs["env_name"] = "Libero_Test_Env"
        data.attrs["macros_image_convention"] = "opengl"
        data.attrs["num_demos"] = 4
        data.attrs["problem_info"] = json.dumps({
            "problem_name": "libero_test",
            "language_instruction": "turn on the stove",
        })
        data.attrs["tag"] = "libero-v1"
        data.attrs["total"] = 4 * NUM_STEPS
        for demo in ("demo_0", "demo_1", "demo_2", "demo_10"):
            _write_demo(data, demo, rng)
