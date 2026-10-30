# MolmoAct2 Observations

These findings come from surveying the source dataset, with a focus on details that aren't covered in the [dataset card](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset).
We're sharing them because they may be useful to others working with the dataset, especially when validating assumptions or building data pipelines around it.

The source revision is [`e9f21ae1`](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/tree/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c).
The survey downloaded 15 file groups, runs of consecutive episodes whose files hold no other episode (see [A note on episodes and files](README.md#a-note-on-episodes-and-files)).
The checks on `meta/` cover every episode.
We also checked the index of all 5,860 video files.

|             | Downloaded | Full dataset |
| ----------- | ---------- | ------------ |
| File groups | 15         | 786          |
| Episodes    | 283        | 32,246       |
| Tasks       | 11         | 34           |
| Size        | 14.8 GB    | 2.35 TB      |
| Robot time  | 4.8 hours  | 704 hours    |

## Summary

`meta/info.json` counts 32,246 episodes, 76,046,658 frames at 30 fps, and 34 tasks.

The frame count works out to 704 hours, while the dataset card mentions more than 720 hours across the full MolmoAct2-BimanualYAM collection.
The recording format is the same in every episode, while the episodes vary widely in length and task.

| Dimension      | Variation    | Notes                                                                                |
| -------------- | ------------ | ------------------------------------------------------------------------------------ |
| Episode length | Diverse      | 23 to 8,953 frames (0.8 s to 5 min), median 2,076 (69 s)                             |
| Video          | Identical    | AV1, 640×360, yuv420p, a keyframe every second frame, for all three cameras          |
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
| `two-demos`  | `episode_31749` | video that ends with part of a discarded take, with no data rows                            |
| `mislabeled` | `episode_07294` | labeled `Pack container`, but shows clothes folded on the table                             |

## Video

An mp4 file in the LeRobot v3 format holds several consecutive episodes.
LeRobot's video encoder writes a keyframe every 2 frames by its [default settings](https://github.com/huggingface/lerobot/blob/a3a8653a5b569e8e70cd6fe4b0973c4b2b983120/src/lerobot/configs/video.py#L86-L92), and this dataset follows them.

In this survey, we looked closely at the episode boundaries, where things can easily go wrong in this format.
See [Some video windows are misaligned](#some-video-windows-are-misaligned).

## Joint values

The joints of `observation.state` (measured) and `action` (commanded) are in radians, and each gripper runs from 0 (closed) to 1 (open).

The measured state follows the command signal by 3 frames (100 ms) in most downloaded episodes (175 of 279), and by 4 or 5 frames in the rest.
The measured joints go past the limits of i2rt's YAM URDF by up to 0.05 rad, and the commanded joints by up to 0.11 rad.

## Episode length and motion

Across the whole dataset, 8 episodes are shorter than 1 s, and 20 are shorter than 3 s.

The per-episode statistics in `meta/episodes/chunk-000/file-000.parquet` show that some episodes have little motion.
Out of all 32,246 episodes, only the right arm moves more than 0.1 rad in 84 episodes, and only the left arm in 30 episodes.
Neither arm moves in 43 episodes.
An arm counts as moving when one of its six joints has a range of `observation.state` from `q01` to `q99` over 0.1 rad.
This cutoff is somewhat arbitrary.
The `no-motion` episode is one of the 43.

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
The dataset card's [Normalization Statistics](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset#normalization-statistics) section says the averaged quantiles are approximate on purpose.

The image statistics and the bookkeeping fields have errors, described under [Edge cases and data bugs](#edge-cases-and-data-bugs).
Joint `std` also reads 0 for 270 joints whose `min` and `max` differ within the episode, so it is unreliable for joints that barely move.

## Edge cases and data bugs

Edge cases are unusual episodes that a pipeline has to handle.
Data bugs and inaccuracies are labels or statistics in the source that are wrong.

### Edge cases

#### Episode 31749's video ends with part of a discarded take

`two-demos` is an episode whose video time windows disagree with its length.
They hold 2,832 frames on `top` and `left` and 2,658 on `right`, but the episode has only 1,619 data rows.
All three cameras cut abruptly at frame 1,619, where the data rows end.
After the cut, the video shows the last 40 s of another take of the same fold.
These frames have no state, action, or task rows.
The likely cause is a LeRobot recording bug, fixed in v0.6.1, that kept a discarded take's frames on disk and encoded them after the re-recorded take ([huggingface/lerobot#3683](https://github.com/huggingface/lerobot/pull/3683)).

#### Episode 7315's top video is one frame short

`videos/observation.images.top/chunk-000/file-490.mp4` holds 32,576 frames, one fewer than the metadata says.
Episode 7315 is the last episode in that file, so it has 4,109 `top` frames for 4,110 data rows.
Its `top` window is also one of the misaligned windows described under [Some video windows are misaligned](#some-video-windows-are-misaligned): it starts one frame late and runs one frame past the end of the file.

### Data bugs and inaccuracies

#### Some video windows are misaligned

`meta/episodes` places each episode in its camera's video file with a time window, from `from_timestamp` to `to_timestamp`.
LeRobot encodes each episode as a separate video segment that starts with a keyframe, and every second frame is a keyframe (see [Video](#video)).
A window that matches its segment therefore starts on a keyframe followed by a delta frame.

We read the index of **all** 5,860 video files, which lists every frame and marks the keyframes.
We then checked every window against those keyframes.
In total, 27 files include 89 misaligned episodes.

| Kind  | Where the window starts                    | Episodes |
| ----- | ------------------------------------------ | -------- |
| late  | on a delta frame                           | 73       |
| early | on a keyframe followed by another keyframe | 18       |
| total |                                            | 89       |

In all 27 affected files, the first misaligned window comes right after a gap in the numbering left over from the merge (see [Statistics left over from the merge](#statistics-left-over-from-the-merge)).
We did not confirm every case, but later episodes in the same file can be misaligned as well.
When a file's last window is misaligned, it can run one frame past the end of the file, as for [episode 7315 above](#episode-7315s-top-video-is-one-frame-short).

Three misalignment examples are illustrated in the diagrams below.
"metadata window" is taken from `meta/episodes`, and "episode's video" is where its own video segment is.

**Episodes 12054 and 12055, top camera (kind: late).**
Visually, episode 12054 ends at 9816, and episode 12055 starts at 9817.
But the metadata window is one frame off and starts episode 12055 at 9818, a delta frame.

![Frames 9815 to 9819 of the top camera: 12055's video starts at 9817, its window at 9818](images/window-12054-12055-top.png)

**Episodes 12053 and 12054, top camera (kind: late).**
Visually, episode 12053 ends at 7342, and episode 12054 starts at 7343, where its metadata window also starts.
But 7343 is a delta frame that can only be decoded from keyframe 7342, so by the keyframes, episode 12054 starts at 7342.

![Frames 7340 to 7344 of the top camera: 12054 starts at 7342 by keyframes and at 7343 by picture](images/window-12053-12054-top.png)

**Episodes 12050 and 12051, left camera (kind: early).**
Visually, episode 12050 ends at 45593, and episode 12051 starts at 45595.
The frame in between, 45594, matches neither of the two adjacent frames.
The metadata window is one frame off and starts episode 12051 at 45594, so it also leaves out episode 12051's last frame.

![Frames 45592 to 45596 of the left camera: a stray frame at 45594 between 12050 and 12051](images/window-12050-12051-left.png)

#### The arms move faster in some episodes

In episodes 24051–24090, such as 24071, the arms seem to move faster than in the episodes before them.
Per frame, the arm joints move about twice as far as in episodes 23782–24050.
The timestamps cannot show whether these episodes were recorded at a different rate, because they appear to be computed from the frame index and the dataset-wide 30 fps.
We found no proof in the data but still report this as a potential issue: some of these episodes might have been recorded at a lower rate than 30 fps.

#### Some `Pack container` episodes show clothes folding

The episodes 7294–7317 are labeled `Pack container`.
In the three we checked (7294, 7295, and 7297), the `top` camera shows clothes folded onto the table, with no container in view.
Several of their instructions still say "place it in the container".
Across the dataset, 52 `Pack container` episodes share their exact instruction with `Clothes Folding` episodes, so the same mislabel likely affects other groups.

#### Image `std` is 0 in every episode

The per-episode image `std` is 0 for every camera in all 32,246 episodes.
The dataset-wide image `std` in `stats.json` (0.017 to 0.037) therefore only measures how much the episode means vary.
The real pixel spread is far wider: the dataset-wide `q01` and `q99` are about 0.01–0.05 and 0.98–1.0.

The likely cause is a LeRobot statistics bug, reported in [huggingface/lerobot#3682](https://github.com/huggingface/lerobot/issues/3682).
It was fixed in v0.6.0 ([huggingface/lerobot#3697](https://github.com/huggingface/lerobot/pull/3697)), but the dataset was compiled before the fix.

#### Statistics left over from the merge

This repository merges many smaller LeRobot datasets, each numbered from 0.
In `meta/episodes`, the statistics of `episode_index`, `index`, and `task_index` still hold those old numbers, while the data rows hold the merged ones.
`stats.json` is combined from the old numbers, so its `episode_index` ends at 129 and its `task_index` is 0 everywhere.

The old numbering starts over 737 times, once per merged dataset.
It also skips 92 numbers in 64 places, and two datasets start at 1, so some source episodes were likely left out of the merge.

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
