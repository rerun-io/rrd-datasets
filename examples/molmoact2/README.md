# MolmoAct2

## Dataset

- **Source**: [allenai/MolmoAct2-BimanualYAM-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) on Hugging Face
- **License**: Apache 2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on three sample file groups (~1.1 GB, 19 episodes) spanning three tasks, all listed in [observations.md](observations.md#surveyed-file-groups).
- **Access**: public, not gated.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

## Local Runs

The source data revision is pinned in this example.
Bump `HF_REVISION` in [`molmoact2/episode_index.py`](molmoact2/episode_index.py) to pick up newer episodes.

### 1. Download

Download the dataset's `meta/` folder (72 MB) and three sample file groups (~1.1 GB, 19 episodes) into `data/MolmoAct2/`.
The dataset stores many episodes in each video file, so the samples are file groups: runs of consecutive episodes whose files hold no other episode.

```bash
pixi run -e molmo download
```

To download different file groups, edit `SAMPLES` in [`molmoact2/download.py`](molmoact2/download.py).

The downloader caches a listing of the whole repo in `.cache/hf_files.json.gz`.
Building it takes a while on the first run, and it is rebuilt when the dataset revision changes.

## Observations

We share our observations including useful details beyond the dataset card.
See [observations.md](observations.md) for the full survey.
