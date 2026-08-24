# HIW-500 Observations

These findings come from surveying a subset of the source episodes, with a focus on details that aren't covered in the [dataset card](https://huggingface.co/datasets/BitRobot/HIW-500).
We're sharing them because they may be useful to others working with the dataset, especially when validating assumptions or building data pipelines around it.

The survey covers 41 episodes: 10 tasks, 25 recording sessions, 34.3 GB, about 89 minutes of robot time.
Note that the full dataset holds 23,000+ episodes.

## Summary

| Dimension            | Feature              | Notes                                                                              |
| -------------------- | -------------------- | ---------------------------------------------------------------------------------- |
| Topic set            | Three variants       | 12-17 topics (growing with the recording date)                                     |
| Frame rates          | Diverse              | 100 Hz (joints) / 50 Hz (Lerobot payload) / 20 Hz (odom) / 10.8–30 Hz (camera)     |
| Resolution and codec | Identical            | all JPEG; head 1280×480 side-by-side stereo, wrist RGB and IR 640×480              |
| Duration and size    | Very diverse         | 14.5 s to 604.6 s; 76 MB to 4.28 GB                                                |
| Calibration sidecars | Not for all episodes | present in the episodes that record IR; head stereo calibration in the newest ones |
| Scene ids            | Four values          | 1 in the older sessions, 7 / 8 / 9 in the newer ones                               |
| Gripper signals      | Two sources          | dex1 joint at 100 Hz and analog LeRobot controls at 50 Hz, on different scales     |

## Example episodes

The table below shows representative converted segments, with one example for each notable case. The `Tag` column provides a short name for referring to each segment in the sections that follow.

| Tag           | Segment id                                                                        | Topics | Calibration  | Notes                                        |
| ------------- | --------------------------------------------------------------------------------- | ------ | ------------ | -------------------------------------------- |
| `sweep-may`   | `Sweep-Floor/episode_2026-05-19_14-35-04/episode_0006`                            | 17     | wrist + head | the slowest head camera (10.8 Hz)            |
| `clothes-may` | `Clothes-Washing/episode_2026-05-25_16-41-06/episode_0002`                        | 17     | wrist + head | the shortest episode (14.5 s), head at 30 Hz |
| `trash-apr`   | `Picking-Trash-To-Rubbish-Bin/episode_2026-04-13_09-30-35/episode_0002`           | 17     | wrist + head | the full rig, nothing missing                |
| `pillow-feb`  | `Move-The-Pillow-To-The-Sofa-From-Floor/episode_2026-02-24_14-31-06/episode_0001` | 13     | none         | no IR streams and no sidecars at all         |
| `clothes-feb` | `Clothes-Washing/episode_2026-02-26_12-11-27/episode_0001`                        | 17     | wrist only   | the left-gripper state payload bug           |
| `kitchen-jan` | `Kitchen-Organization/episode_2026-01-28_11-27-45/episode_0001`                   | 12     | none         | the oldest variant, no `/wbc_lerobot`        |

## Topic set — three variants, growing by recording date

| Topics | Delta                                                        | When                       | Example       |
| ------ | ------------------------------------------------------------ | -------------------------- | ------------- |
| 12     | the core set                                                 | the oldest sessions        | `kitchen-jan` |
| 13     | plus `/wbc_lerobot` (50 Hz)                                  | from late January          | `pillow-feb`  |
| 17     | plus `/camera/{left,right}_wrist/ir{1,2}/compressed` (30 Hz) | from late February onwards | `sweep-may`   |

The rig gained capabilities over time, and the boundaries are not clean day cuts: sessions recorded on the same February day fall on either side of the IR boundary.
The oldest episodes have no end-effector observations or actions at all, only joint-space `lowstate` / `lowcmd` and the gripper channels.
Any pipeline that needs `/wbc_lerobot` has to either skip those episodes or derive end-effector poses by forward kinematics.

## Head-camera rate varies

Head camera rate varies between 10.8 and 30 Hz while wrist cameras (both RGB and IR) stay at 29.9–30 Hz throughout.
Frame gaps are quantized to multiples of ~33 ms, and the skip pattern differs per episode:

```
~15 Hz    gaps {1×33ms: 5, 2×33ms: 584}          steady every 2nd frame
~13 Hz    gaps {1: 13, 2: 1274, 3: 1190, 4: 1}   irregular 2–3 frame skips
~10.8 Hz  gaps {2: 5, 3: 173}                    steady every 3rd frame
~30 Hz    gaps {1: 592, 2: 1}                    no skipping
```

## Calibration sidecars

The IR-recording episodes ship `calibration/params/camera_<serial>.json`, which are RealSense-style files with color, ir1 and ir2 intrinsics plus IR-to-color extrinsics.
Distinct wrist-serial pairs recur across episodes.

The newest episodes also carry `head_camera_params.yaml`, a full OpenCV stereo calibration of the head pair (640×480 per eye, baseline ≈ 60.5 mm).
The oldest episodes have no head intrinsics.

Three consequences for anyone reading these files:

- Treat calibration as per-session data, never as a dataset constant.
- A wrist file cannot stand in for the head: its focal length runs about 36% above the head's per-eye value, and its IR baseline of 18.1 mm is a third of the head's 60.5 mm.
- The serial-to-side mapping is not derivable from the filename, only from the calibration content.

## Gripper signals — two sources that do not agree

Every episode carries the gripper twice, at different rates and on different scales.

| Source                              | Rate   | Converted entities                | Observed range |
| ----------------------------------- | ------ | --------------------------------- | -------------- |
| `/stamped/dex1/<side>/state`        | 100 Hz | `/state/gripper/<side>/q`         | 0.02 .. 4.62   |
| `/stamped/dex1/<side>/cmd`          | 100 Hz | `/cmd/gripper/<side>/q`           | 0.02 .. 4.50   |
| `/wbc_lerobot` → `gripper_controls` | 50 Hz  | `/lerobot/gripper/<side>_trigger` | 0 .. 10        |
| `/wbc_lerobot` → `gripper_controls` | 50 Hz  | `/lerobot/gripper/<side>_squeeze` | 0 .. 1         |

Ranges are from a set of selected episodes, so treat them as indicative rather than dataset-wide.
We didn't find a document explaining `/wbc_lerobot`.

**One episode is missing a signal entirely.**
The February Clothes-Washing session's left `state` channel carries command payloads (see the data-bugs section), so its converted output has seven gripper entities instead of eight — no `/state/gripper/left/q` at all.

Viewing consequence: the two sources cannot share a plot axis, since a `trigger` at 10 flattens a `q` near 1.
The default blueprint gives the gripper one slot with a `Dex1` tab (measured against commanded) and a `LeRobot` tab (trigger and squeeze).

## Duration, size, and subtasks

```
duration:  14.5 s  to 604.6 s   median  69.3 s
size:      76 MB   to 4.28 GB   median  519 MB
messages:  14,182  to 592,220   median  62,933
subtasks:  2 to 30 per episode, around 70 distinct labels
```

The newer episodes trend shorter, and each new batch of sessions adds fresh subtask vocabulary.
The label set is open-ended and grows with scale.

Files are uncompressed MCAP, so anything that rewrites chunks shrinks storage substantially.

## Data bugs and edge cases

### `/stamped/dex1/left/state` carries command payloads

In one February Clothes-Washing session including `clothes-feb`, every message on the left gripper state channel is serialized as `unitree_go/MotorCmd` and numerically identical to the concurrent `/stamped/dex1/left/cmd` messages. The channel still declares `MotorStateStamped`, which is 12 bytes larger, so strict CDR decoders throw.

The `.rrd` base layer compares decoded rows against the MCAP's own per-channel counts and records `has_undecodable` and `undecodable_topics` in `property:episode`.

## Base-layer entity paths

The source carries more signals than one layout can show, and the base layer keeps all of them.
The default blueprint plots the few an episode is usually opened for, so the rest sits in the recording unseen.
The table lists every entity and marks which ones that layout shows, so a reader can find what is there as well as what is plotted.

The blueprint column names the view an entity lands in, or `—` when the default layout does not show it.
Sibling paths are collapsed with braces; expanded, the sample holds 66 entities.

| Entity path                                                            | Type                                                     | Default blueprint | Source                               |
| ---------------------------------------------------------------------- | -------------------------------------------------------- | ----------------- | ------------------------------------ |
| `/__properties/episode`                                                | `has_undecodable`, `undecodable_topics` (static)         | —                 | channel census                       |
| `/annotation`                                                          | `TextDocument`                                           | —                 | `/annotation`                        |
| `/calibration/params/{head_camera_params,wrist_camera1,wrist_camera2}` | `CalibrationFile` (static)                               | —                 | `calibration/` sidecars              |
| `/camera/head/{left,right}`                                            | `EncodedImage`                                           | Head L, Head R    | `/camera/head/image/compressed`      |
| `/camera/{left,right}_wrist/image/compressed`                          | `CoordinateFrame`, `EncodedImage`                        | Wrist L, Wrist R  | same topic                           |
| `/cmd/gripper/<side>`                                                  | `kp`, `kd` (static)                                      | —                 | `/stamped/dex1/<side>/cmd`           |
| `/cmd/gripper/<side>/q`                                                | `Scalars`                                                | Gripper(Dex1)     | `/stamped/dex1/<side>/cmd`           |
| `/cmd/joint`                                                           | `joint_names`, `kp`, `kd` (static)                       | —                 | `/stamped/lowcmd`                    |
| `/cmd/joint/{q,tau}`                                                   | `Scalars`                                                | —                 | `/stamped/lowcmd`                    |
| `/episode`                                                             | `task`, `scene`, `duration_sec`, `episode_name` (static) | —                 | `info.json` sidecar                  |
| `/lerobot/ee_{state,action}`                                           | `Scalars`, `ee_names` (static)                           | End-effector      | `/wbc_lerobot`                       |
| `/lerobot/ee_{state,action}/{left,right}`                              | `Transform3D`                                            | End-effector      | `/wbc_lerobot`                       |
| `/lerobot/gripper/{left,right}_{squeeze,trigger}`                      | `Scalars`                                                | Gripper(LeRobot)  | `/wbc_lerobot`                       |
| `/lerobot/pivot/0`…`/lerobot/pivot/6`                                  | `Scalars`                                                | —                 | `/wbc_lerobot`                       |
| `/state/base`                                                          | `Transform3D`                                            | Scene             | `/lf/odommodestate`                  |
| `/state/base/{body_height,yaw_speed}`                                  | `Scalars`                                                | —                 | `/lf/odommodestate`                  |
| `/state/base/{position,velocity}/{x,y,z}`                              | `Scalars`                                                | —                 | `/lf/odommodestate`                  |
| `/state/gripper/<side>/q`                                              | `Scalars`                                                | Gripper(Dex1)     | `/stamped/dex1/<side>/state`         |
| `/state/gripper/<side>/{dq,tau}`                                       | `Scalars`                                                | —                 | `/stamped/dex1/<side>/state`         |
| `/state/imu/pelvis/{rpy,gyroscope,accelerometer}`                      | `Scalars`, `imu_names` (static)                          | —                 | `/stamped/lowstate`                  |
| `/state/imu/secondary/{rpy,gyroscope,accelerometer}`                   | `Scalars`, `imu_names` (static)                          | —                 | `/stamped/secondary_imu`             |
| `/state/imu/{pelvis,secondary}/temperature`                            | `Scalars`                                                | —                 | `lowstate`, `/stamped/secondary_imu` |
| `/state/joint`                                                         | `joint_names` (static)                                   | —                 | `/stamped/lowstate`                  |
| `/state/joint/{q,dq,tau}`                                              | `Scalars` (width 29)                                     | Joints            | `/stamped/lowstate`                  |
| `/state/joint/{voltage,status,mode}`                                   | `Scalars` (width 29)                                     | —                 | `/stamped/lowstate`                  |
| `/state/joint/temperature/{0,1}`                                       | `Scalars` (width 29)                                     | —                 | `/stamped/lowstate`                  |
| `/task/subtask`                                                        | `StateChange`                                            | Subtasks          | `info.json` sidecar                  |

Twenty-four of the 66 reach the default blueprint.
The rest are static metadata, motor health, base-motion scalars, and the LeRobot pivot channels.

The layout also declares four wrist-IR views and pulls `/robot/**` and `/odom/**` into the 3D scene.
Those entities live in the `ir`, `urdf` and `odom` layers, so they are absent from a base-only recording.
