# LIBERO Dataset

These findings come from surveying five task files, one per suite.

## Summary

The source files are uniform, as expected of simulated data: the same item tree and attribute keys everywhere.

### Source hdf5 layout

```
<suite>/<task>_demo.hdf5              # one file per task, exactly 50 demos in every surveyed file
└── data                              # attrs: bddl_file_name, env_args, env_name,
    │                                 #   macros_image_convention ("opengl"), num_demos,
    │                                 #   problem_info, tag ("libero-v1"), total
    ├── demo_0                        # attrs: init_state [f64: N], model_file (MuJoCo XML),
    │   │                             #   num_samples (= N, the demo's step count)
    │   ├── actions                   [f64: N × 7]
    │   ├── dones                     [u8:  N]
    │   ├── obs
    │   │   ├── agentview_rgb         [u8:  N × 128 × 128 × 3]
    │   │   ├── ee_ori                [f64: N × 3]
    │   │   ├── ee_pos                [f64: N × 3]
    │   │   ├── ee_states             [f64: N × 6]
    │   │   ├── eye_in_hand_rgb       [u8:  N × 128 × 128 × 3]
    │   │   ├── gripper_states        [f64: N × 2]
    │   │   └── joint_states          [f64: N × 7]
    │   ├── rewards                   [u8:  N]
    │   ├── robot_states              [f64: N × 9]
    │   └── states                    [f64: N × 47…110]
    └── demo_1 …                      # same shape, N varies per demo
```

- The task description is found at `problem_info.language_instruction`.
- The source does store simulation time: column 0 of the `states` item (`states[:, 0]`), MuJoCo's internal clock.
- Items within a demo are step-aligned: one entry per 20 Hz control step, `num_samples` in total — the per-demo alignment `Hdf5Reader` needs.

## Example files

The five files `download` fetches, one per suite:

| Suite          | Task                                                                                 | Size    | Demonstrates                                               |
| -------------- | ------------------------------------------------------------------------------------ | ------- | ---------------------------------------------------------- |
| libero_10      | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`                        | 1.32 GB | long-horizon composite task, longest demos (219–340 steps) |
| libero_90      | `KITCHEN_SCENE3_turn_on_the_stove`                                                   | 0.46 GB | the first subtask of the libero_10 sample, same scene      |
| libero_goal    | `turn_on_the_stove`                                                                  | 0.45 GB | the same behavior again in a third suite                   |
| libero_object  | `pick_up_the_salad_dressing_and_place_it_in_the_basket`                              | 0.66 GB | the floor scene, and the one shared-`model_file` file      |
| libero_spatial | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 0.51 GB | spatial-relation task                                      |

## Redundancies

Every dataset is in the `.rrd`; the ones below repeat others and are not plotted by default.

- `rewards` and `dones` are byte-identical in every surveyed demo.
- `init_state` equals `states[0]`, so the demo attribute already carries the initial state.
- `states` (47–110 floats, scene-dependent width) is the raw MuJoCo state vector: `[sim_time, qpos, qvel, …]` with no per-file schema.
  It only means something to a robosuite reconstruction, which `model_file` + `init_state` serve better.
  Its first column is the simulation clock, which the `sim_time` timeline reproduces.
- `robot_states` is exactly `[gripper_states (2), ee_pos (3), ee_quat_xyzw (4)]`, verified numerically.
- `ee_states` is exactly `[ee_pos, ee_ori]` concatenated.

## Other notes on source data

- `macros_image_convention` is `opengl` in all five files: raw frames render the scene upside-down ([robosuite stores OpenGL-convention buffers](https://robosuite.ai/)).
- The axis-angle (`ee_ori`) magnitude can exceed π — the vector tracks the rotation continuously instead of wrapping.
- Demo keys are contiguous from `demo_0` in every surveyed file, but HDF5 iterates them lexicographically (`demo_0, demo_1, demo_10, …`) — sort numerically when order matters.
- Each LIBERO task is specified by a .bddl file (Behavior Domain Definition Language) and the HDF5 records where that file lives twice: `env_args.env_kwargs.bddl_file_name` and `bddl_file_name` both as /data attribute.
