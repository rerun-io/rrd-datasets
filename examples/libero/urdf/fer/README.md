# Franka FER (Panda) URDF

A plain-URDF export of the [franka_description](https://github.com/frankaemika/franka_description) `fer` model, vendored for the LIBERO URDF layer.
Apache-2.0 — the upstream `LICENSE` and `NOTICE` ride alongside.

The model: root link `base`, revolute `fer_joint1`…`fer_joint7`, prismatic `fer_finger_joint1`/`2`, and the hand mounted at the flange with the stock `0 0 -pi/4` offset.
LIBERO's `obs/joint_states[i]` maps to `fer_joint{i+1}`.

## Regenerating from source

1. Clone franka_description at commit `7aeeddc449edf8d62b594f9e36a81da53e7796f9` (2026-08-17).
2. Inline `$(find franka_description)` in every `*.xacro` with the checkout path — pip's standalone `xacro` has no ROS package index to resolve it.
3. Expand with `xacro` 2.1.1: `xacro robots/fer/fer.urdf.xacro include_self_collision_geometry:=false > fer_raw.urdf`.
4. Trim for vendoring: drop every `<collision>` element, drop the empty `*_accelerometer_*` mount links and the fixed joints holding them, rewrite `package://franka_description/` mesh paths to relative, and copy only the referenced visual `.dae` meshes.
