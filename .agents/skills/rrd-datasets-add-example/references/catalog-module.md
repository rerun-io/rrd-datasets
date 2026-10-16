# Catalog module template

`catalog.py` registers one dataset from the `.rrd` files already on disk: one segment per episode, one named layer per file, plus the shared asset and the default blueprint.
The `rerun-dataset-conversion` skill's `references/registering.md` describes the API and its rules.
This file is only the module template.

## Rules for the module

Take the recording ids from the base layer's files, so a layer that skipped an episode cannot invent a segment of its own.
Report a missing layer file by name instead of registering around it silently, since a segment missing a layer opens in the viewer with no error.
Say what to run when the asset or the blueprint is absent, rather than failing on it.
Default every argument so a bare run works, and offer `--recreate` for a rebuild from scratch.

---

```python
"""
Register the converted [Dataset Name] RRDs into a Rerun catalog as one dataset.

Each episode is one segment, keyed by its recording id; its `.rrd` files attach as the
layers named in `LAYERS`. The shared model registers as an asset, and the default
blueprint is installed on the dataset.

Run:  pixi run serve                  # start the catalog, leave it running
      pixi run -e [env] register      # register every converted episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from [pkg].blueprint import BLUEPRINT_PATH
from [pkg].layers import LAYERS
from [pkg].urdf_layer import model_rrd_path
from rrd_datasets_common.paths import dataset_rrd_dir, layer_relpath

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"
DATASET_NAME = "[dataset]"


def recording_ids(rrd_dir: Path) -> list[str]:
    """One id per episode, taken from the base layer's files."""
    return sorted(path.stem for path in (rrd_dir / "base").glob("*.rrd"))


def register_episodes(
    catalog_url: str,
    dataset_name: str,
    rrd_dir: Path,
    blueprint: Path,
    *,
    recreate: bool = False,
) -> DatasetEntry:
    """Register each episode's layers as one segment, attach the shared asset, and set the blueprint."""
    ids = recording_ids(rrd_dir)
    if not ids:
        raise FileNotFoundError(f"No base *.rrd files in {rrd_dir}")

    client = CatalogClient(catalog_url)
    if recreate and dataset_name in client.dataset_names():
        client.get_dataset(dataset_name).delete()
    dataset = client.create_dataset(dataset_name, exist_ok=True)

    for layer in LAYERS:
        paths = [rrd_dir / layer_relpath(layer, rec_id) for rec_id in ids]
        uris = [path.resolve().as_uri() for path in paths if path.exists()]
        if not uris:
            print(f"  layer '{layer}': no files, skipping")
            continue
        dataset.register(uris, layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        print(f"  layer '{layer}': registered {len(uris)} file(s)")
        for missing in (path for path in paths if not path.exists()):
            print(f"    missing: {missing.name} — this episode registers without its '{layer}' layer")

    model = model_rrd_path(rrd_dir)
    if model.exists():
        dataset.register_asset(model.resolve().as_uri())
    else:
        print(f"  no model rrd at {model} (run `pixi run -e [env] convert-urdf`)")

    if blueprint.exists():
        dataset.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
    else:
        print(f"  no blueprint at {blueprint} (run `pixi run -e [env] blueprint`)")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Register [Dataset Name] RRDs into a Rerun catalog.")
    parser.add_argument("--rrd-dir", type=Path, default=dataset_rrd_dir(DATASET_NAME))
    parser.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL, help="Catalog gRPC URL (`pixi run serve`).")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--blueprint", type=Path, default=BLUEPRINT_PATH)
    parser.add_argument("--recreate", action="store_true", help="Delete an existing dataset first.")
    args = parser.parse_args()

    register_episodes(
        args.catalog_url, args.dataset_name, args.rrd_dir, args.blueprint, recreate=args.recreate
    )


if __name__ == "__main__":
    main()
```
