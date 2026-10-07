# MolmoAct2 Observations

These findings come from surveying a subset of the source episodes, with a focus on details that aren't covered in the [dataset card](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset).
We're sharing them because they may be useful to others working with the dataset, especially when validating assumptions or building data pipelines around it.

The source revision is [`e9f21ae1`](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/tree/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c).
The survey downloaded 15 file groups, runs of consecutive episodes whose files hold no other episode (see [Episodes share files](#episodes-share-files)).
The checks on `meta/` cover every episode.

|             | Surveyed  | Full dataset |
| ----------- | --------- | ------------ |
| File groups | 15        | 786          |
| Episodes    | 283       | 32,246       |
| Tasks       | 11        | 34           |
| Size        | 14.8 GB   | 2.35 TB      |
| Robot time  | 4.8 hours | 704 hours    |

## Summary

`meta/info.json` counts 32,246 episodes, 76,046,658 frames at 30 fps, and 34 tasks.

The frame count works out to 704 hours, while the dataset card mentions more than 720 hours across the full MolmoAct2-BimanualYAM collection.
The recording format is the same in every episode, while the episodes vary widely in length and task.

| Dimension      | Variation    | Notes                                                                                |
| -------------- | ------------ | ------------------------------------------------------------------------------------ |
| Episode length | Diverse      | 23 to 8,953 frames (0.8 s to 5 min), median 2,076 (69 s)                             |
| Video          | Identical    | AV1, 640×360, yuv420p, a keyframe every second frame, for all three cameras          |
| Calibration    | None         | no robot model, no camera intrinsics or extrinsics                                   |
| Tasks          | 34 strings   | 32 after ignoring case and trailing dots, plus one free-text instruction per episode |
| File layout    | Shared files | a median of 10 episodes per data file and 12 to 19 per video file                    |

## Source layout

```
allenai/MolmoAct2-BimanualYAM-Dataset/
├── meta/
│   ├── info.json                         # features, fps, file path templates; robot_type bi_yam_follower
│   ├── stats.json                        # dataset-wide statistics of every feature
│   ├── tasks.parquet                     # the 34 task strings and their task_index
│   ├── tasks_annotated.parquet           # one free-text instruction per episode_index
│   └── episodes/chunk-000/file-000.parquet
│                                         # one row per episode: length, task, file locations,
│                                         #   video time windows, statistics of every feature
├── data/chunk-NNN/file-NNN.parquet       # 3,576 files, one row per frame: action [14],
│                                         #   observation.state [14], timestamp, frame_index,
│                                         #   episode_index, index, task_index
└── videos/observation.images.<camera>/chunk-NNN/file-NNN.mp4
                                          # top (2,488 files), left (1,765), right (1,607);
                                          #   consecutive episodes back to back
```

## Example episodes

The table below shows representative episodes, with one example for each notable case.
The `Tag` column provides a short name for referring to each episode in the sections that follow.

| Tag          | Segment id      | Demonstrates                                                                                |
| ------------ | --------------- | ------------------------------------------------------------------------------------------- |
| `blocks`     | `episode_00807` | four letter blocks picked out of a scattered set and placed in order (`spell out CVPR`)     |
| `snacks`     | `episode_01829` | six named snacks packed into a box (`Pack container`)                                       |
| `cloth-fold` | `episode_31673` | one cloth folded onto another, in a file group of its own (`Fold the cloth`)                |
| `untangle`   | `episode_31487` | both arms pulling cables apart, with the right gripper closing 17 times (`Untangle cables`) |
| `dishwasher` | `episode_24832` | a five-step instruction: bowls and a plate into the rack, food waste into the bin           |
| `no-motion`  | `episode_01832` | 82 frames in which no joint moves                                                           |
| `two-demos`  | `episode_31749` | video that holds a second demonstration with no data rows                                   |
| `mislabeled` | `episode_07294` | labeled `Pack container`, but shows clothes folded on the table                             |

## Episodes share files

Each data parquet holds a median of 10 consecutive episodes, and each mp4 a median of 12 to 19 depending on the camera, back to back.
The three cameras start new files at different episodes, so the files of one episode also hold parts of its neighbors.

Because neighboring episodes share files, the episodes fall into 786 file groups: runs of consecutive episodes whose files hold no other episode.
A group holds a median of 40 episodes, at most 130, and about 3 GB.
Most groups match one of the smaller datasets that were merged into this repository (see [Statistics left over from the merge](#statistics-left-over-from-the-merge)).
The exception is the `Fold the cloth` dataset with episodes 31647–31696, which stored every episode in its own files, so each of its 50 episodes is a group of its own, such as `cloth-fold`.

## Video

Every second frame is a keyframe in every surveyed video, so every episode window starts on a keyframe.

## Joint values

The joints of `observation.state` (measured) and `action` (commanded) are in radians, and each gripper runs from 0 (closed) to 1 (open).

The measured state follows the command by 3 frames (100 ms) in most surveyed episodes (175 of 279), and by 4 or 5 frames in the rest.
The measured joints go past the limits of i2rt's YAM URDF by up to 0.05 rad, and the commanded joints by up to 0.11 rad.

## Episode length and motion

8 episodes are shorter than 1 s, and 20 are shorter than 3 s.

Judging by the per-episode joint ranges, some episodes have little motion.
In 84 episodes only the right arm moves more than 0.1 rad, in 30 only the left arm, and in 43 neither arm.
`no-motion` is one of the 43.

## Tasks and instructions

Each episode has one task, from 34 task strings.
The source spells some tasks in several ways: `Spell out Ai2` and `spell out AI2`, `Clothes Folding` and `Folding CLothes`, and three variants of `Sort waste into correct bins and load items into the dishwasher`, with and without `rack` and a trailing dot.
Ignoring case and trailing dots leaves 32 distinct tasks.
The variants that remain share many instructions, so they are likely the same tasks too.

`meta/tasks_annotated.parquet` holds one free-text instruction per episode, more specific than the task, e.g. `Fold jeans.` under `Clothes Folding`.
There are 11,451 distinct instructions, and none is equal to its task.
Some tasks reuse a few phrasings: `Plug charger to phone and switch on socket` has one distinct instruction per 16 episodes.
The full task list is in the [appendix](#appendix-task-list).

## Statistics

The dataset-wide statistics in `meta/stats.json` are combined from the per-episode statistics in `meta/episodes`.
`min` and `max` are exact, while `mean` and the quantiles are averages of the per-episode values, weighted by frame count.

The image statistics and the bookkeeping fields have errors, described under [Edge cases and data bugs](#edge-cases-and-data-bugs).
Joint `std` also reads 0 for 270 joints whose `min` and `max` differ within the episode, so it is unreliable for joints that barely move.

## Edge cases and data bugs

### Episode 31749's video holds a second demonstration

The time windows of `two-demos` are longer than its 1,619 data rows: 2,832 frames on `top` and `left`, and 2,658 on `right`.
The extra video shows a second, complete fold of the same cloth, which has no state, action, or task rows.
The video runs 40 s past the data on `top` and `left`, and the `right` window ends 174 frames (5.8 s) earlier than the other two.
It is the only episode whose windows disagree with its length.

### Episode 7315's top video is one frame short

`videos/observation.images.top/chunk-000/file-490.mp4` holds 32,576 frames, one fewer than the metadata says.
Episode 7315 is the last episode in that file, so it has 4,109 `top` frames for 4,110 data rows.

### Some `Pack container` episodes show clothes folding

The episodes 7294–7317 are labeled `Pack container`.
In the three we checked (7294, 7295, and 7297), the `top` camera shows clothes folded onto the table, with no container in view.
Several of their instructions still say "place it in the container".
Across the dataset, 52 `Pack container` episodes share their exact instruction with `Clothes Folding` episodes, so the same mislabel likely affects other groups.

### Image `std` is 0 in every episode

The per-episode image `std` is 0 for every camera in all 32,246 episodes.
The dataset-wide image `std` in `stats.json` (0.017 to 0.037) therefore only measures how much the episode means vary.
The real pixel spread is far wider: the dataset-wide `q01` and `q99` are about 0.01–0.05 and 0.98–1.0.

### Statistics left over from the merge

This repository merges many smaller LeRobot datasets.
The per-episode statistics of `episode_index`, `index`, and `task_index` still hold the values from before the merge: each source dataset numbered its episodes and frames from 0, and used task 0.
Only the first 16 episodes match their merged values.
The data rows hold the correct merged values.
The dataset-wide values in `stats.json` are meaningless for the same reason: `episode_index` ends at 129, and `task_index` is 0 everywhere.

The old numbering marks 737 merged datasets, since it starts over at 0 or 1 wherever a new one begins.
It skips 92 numbers in 64 places, and two datasets start at 1, which suggests some source episodes were left out of the merge.

## Surveyed file groups

The three groups marked `SAMPLES` are the ones `pixi run -e molmo download` fetches.
The others were downloaded for the survey.
The file columns give each group's files as `chunk/file`.
For example, `003/527–532` under `data` stands for `data/chunk-003/file-527.parquet` through `file-532.parquet`, and `002/462` under `top` stands for `videos/observation.images.top/chunk-002/file-462.mp4`.

| First episode | Episodes | Task                                                                   | Size    | `data`        | `top`         | `left`        | `right`       | Chosen for                                              |
| ------------- | -------- | ---------------------------------------------------------------------- | ------- | ------------- | ------------- | ------------- | ------------- | ------------------------------------------------------- |
| 800           | 10       | `spell out CVPR`                                                       | 0.66 GB | `000/089`     | `000/062`     | `000/052`     | `000/046`     | `SAMPLES`; `blocks`                                     |
| 890           | 10       | `form a tower`                                                         | 0.58 GB | `000/098`     | `000/071`     | `000/061`     | `000/054`     | task coverage                                           |
| 1825          | 8        | `Pack container`                                                       | 0.37 GB | `000/209–210` | `000/141`     | `000/119`     | `000/109`     | `SAMPLES`; `snacks`, `no-motion`, a bright right camera |
| 2479          | 20       | `Pack container`                                                       | 1.10 GB | `000/311–313` | `000/182`     | `000/154`     | `000/142`     | a large joint peak in the per-episode statistics        |
| 2892          | 20       | `Plug charger to phone and switch on socket`                           | 0.73 GB | `000/357–358` | `000/207`     | `000/170`     | `000/160`     | task coverage                                           |
| 7294          | 24       | `Pack container`                                                       | 1.90 GB | `000/829–832` | `000/489–491` | `000/358–359` | `000/353–354` | `mislabeled`, episode 7315's short `top` video          |
| 11726         | 17       | `Clothes Folding`                                                      | 0.81 GB | `001/306–309` | `000/801`     | `000/575`     | `000/563`     | a short episode, command and state far apart            |
| 13174         | 20       | `Clean dirty plates on tray and stack at the side`                     | 0.68 GB | `001/460–461` | `000/886`     | `000/634`     | `000/619`     | a dark left camera                                      |
| 18215         | 20       | `scan the item barcode`                                                | 1.73 GB | `002/035–036` | `001/240–241` | `000/899–900` | `000/815`     | a large joint peak in the per-episode statistics        |
| 20981         | 20       | `Scoop and weigh 50g on scale`                                         | 1.18 GB | `002/346–347` | `001/548`     | `001/122`     | `000/989`     | task coverage                                           |
| 24814         | 20       | `Sort waste into correct bins and load items into the dishwasher rack` | 1.01 GB | `002/740–741` | `001/881`     | `001/308`     | `001/180`     | `dishwasher`                                            |
| 28126         | 33       | `Pack container`                                                       | 0.69 GB | `003/087–093` | `002/111`     | `001/455`     | `001/317`     | the shortest episode, joints that never move            |
| 31487         | 10       | `Untangle cables`                                                      | 0.60 GB | `003/455`     | `002/391`     | `001/675`     | `001/517`     | `untangle`                                              |
| 31673         | 1        | `Fold the cloth`                                                       | 0.04 GB | `003/497`     | `002/436`     | `001/716`     | `001/559`     | `SAMPLES`; `cloth-fold`                                 |
| 31746         | 50       | `Fold the cloth`                                                       | 2.75 GB | `003/527–532` | `002/462–463` | `001/742–743` | `001/585–586` | `two-demos`                                             |

## Appendix: task list

The tasks in order of their first episode, with their size.

| Task                                                                    | Episodes | Hours |
| ----------------------------------------------------------------------- | -------- | ----- |
| `Spell out Ai2`                                                         | 160      | 2.2   |
| `spell out AI2`                                                         | 70       | 1.4   |
| `spell out CORL`                                                        | 72       | 1.7   |
| `spell out ROBOT`                                                       | 80       | 2.3   |
| `spell out RSS`                                                         | 76       | 1.2   |
| `spell out ICRA`                                                        | 60       | 1.5   |
| `spell out CVPR`                                                        | 30       | 0.5   |
| `form a horizontal line`                                                | 159      | 4.4   |
| `form a circle`                                                         | 112      | 3.3   |
| `form a tower`                                                          | 111      | 2.7   |
| `spell out NEURIPS`                                                     | 60       | 2.6   |
| `lift 3 blocks`                                                         | 46       | 1.1   |
| `lift 4 blocks`                                                         | 40       | 1.2   |
| `lift 5 blocks`                                                         | 40       | 1.2   |
| `rotate 3 blocks`                                                       | 90       | 0.8   |
| `rotate 4 blocks`                                                       | 80       | 0.9   |
| `rotate 5 blocks`                                                       | 59       | 1.1   |
| `flip 3 blocks`                                                         | 40       | 1.1   |
| `flip 4 blocks`                                                         | 40       | 1.1   |
| `flip 5 blocks`                                                         | 50       | 1.6   |
| `Pack container`                                                        | 7,041    | 123.7 |
| `Plug charger to phone and switch on socket`                            | 2,596    | 51.4  |
| `Clothes Folding`                                                       | 3,862    | 86.3  |
| `Folding CLothes`                                                       | 377      | 5.8   |
| `Clean dirty plates on tray and stack at the side`                      | 3,891    | 71.2  |
| `scan the item barcode`                                                 | 3,053    | 103.6 |
| `Scoop item into bowl and weigh 50g on scale`                           | 503      | 14.3  |
| `Scoop item into bowl and weigh 150g on scale`                          | 466      | 15.8  |
| `Scoop and weigh 50g on scale`                                          | 80       | 1.8   |
| `Sort waste into correct bins and load items into the dishwasher rack`  | 2,540    | 49.6  |
| `Sort waste into correct bins and load items into the dishwasher rack.` | 1,110    | 15.2  |
| `Sort waste into correct bins and load items into the dishwasher`       | 1,763    | 36.1  |
| `Untangle cables`                                                       | 2,890    | 84.7  |
| `Fold the cloth`                                                        | 599      | 10.8  |
