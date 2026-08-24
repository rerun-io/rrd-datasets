# LIBERO

[LIBERO](https://libero-project.github.io/) is a lifelong-learning manipulation benchmark simulated in [robosuite](https://robosuite.ai/)-based environments: 130 tabletop tasks across five suites, each task shipping ~50 teleoperated demos as one HDF5 file.
This example converts each demo into Rerun recordings (`.rrd`).

**Status: incubating.** The conversion is under construction; only the sample download below works today.

> **Note:** this example uses Pixi. Get it [here](https://pixi.prefix.dev/latest/installation/).
> Everything runs inside the pixi env: prefix task commands with `pixi run`.
> File paths in the commands below are relative to the repository root.

## Dataset

- **Source**: [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) on Hugging Face
- **License**: Apache 2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Revision**: [`f13aa24a`](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a), dated 2025-05-18, pinned by the converter.
- **Subset used**: the local demo runs on five sample task files (~3.4 GB), one per suite.
- **Access**: public, no gating.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

## Local Runs

### 1. Download

```bash
pixi run -e libero download
```

Fetches the five sample task files into `data/LIBERO/<suite>/`.
