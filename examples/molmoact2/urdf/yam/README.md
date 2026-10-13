# YAM URDF

The YAM v1 arm from [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt), vendored unchanged for the MolmoAct2 URDF layer.
MIT — the upstream `LICENSE` is next to it.

The files are `i2rt/robot_models/arm/yam/v1/yam.urdf` and the nine meshes it references, at commit `120c3c81400171174604e503943f8d1ebc891058` (2026-09-17).
Its gripper meshes are the ones i2rt's `linear_4310` station model uses, the gripper type MolmoAct2's own YAM driver selects (`examples/yam/gello_min/yam.py` in [allenai/molmoact2](https://github.com/allenai/molmoact2)).

The model: root link `base`, revolute `joint1` to `joint6`, and the prismatic finger joints `joint7` and `joint8` on the `gripper` flange.

## Mapping from MolmoAct2

- `{side}_joint_{i}.pos` drives `joint{i+1}`, in radians, with no offset or sign change.
- `{side}_gripper.pos` runs from 0 (closed) to 1 (open), and both finger joints take `-0.04695 × {side}_gripper.pos` metres.

The finger meshes touch at a joint value of 0, so 0 is the closed end of the `[-0.04695, 0]` range.
The finger link frames move toward each other as the fingers open, so check the direction against the meshes, not the frames.
