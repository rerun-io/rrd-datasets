"""
Register the per-episode HIW-500 RRDs into a local Rerun catalog as one dataset.

`rerun server -d rrds/hiw-500` fails here ("Layer 'base' already exists") because every `.rrd` is
loaded under the default layer name `base`, and each episode has several files sharing one
`recording_id`. Layer names are assigned at *registration* time, not when writing the RRD.

So the flow mirrors the cloud ingestion pattern, with two pixi tasks:

    pixi run serve           # start the in-memory Rerun catalog (leave running)
    pixi run -e hiw register    # register every episode's layers into a dataset

Each episode is one *segment* (id = the shared `recording_id`); its RRDs attach as the `base`,
`urdf`, `odom`, `cameras`, `ir`, and `properties` *layers* of that segment, and a default
blueprint is installed on the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from hiw_500.blueprint import BLUEPRINT_PATH
from hiw_500.layers import LAYERS
from rrd_datasets_common.paths import dataset_rrd_dir, layer_relpath

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"
DATASET_NAME = "hiw-500"


def _base_rrds(rrd_dir: Path) -> list[Path]:
    """One base recording per episode: the `base/<recording_id>.rrd` files under `rrd_dir`."""
    return sorted((rrd_dir / "base").glob("*.rrd"))


def register_episodes(
    catalog_url: str,
    dataset_name: str,
    rrd_dir: Path,
    blueprint: Path,
    *,
    recreate: bool = False,
) -> DatasetEntry:
    """
    Register each episode's layers as one segment of the dataset and set the blueprint.

    The dataset is created if missing, and layers that are already there are replaced.
    With `recreate`, an existing dataset of the same name is deleted first and rebuilt from scratch.
    """
    base_rrds = _base_rrds(rrd_dir)
    if not base_rrds:
        raise FileNotFoundError(f"No base *.rrd files found in {rrd_dir}")

    client = CatalogClient(catalog_url)
    if recreate and dataset_name in client.dataset_names():
        client.get_dataset(dataset_name).delete()
    dataset = client.create_dataset(dataset_name, exist_ok=True)

    on_duplicate = OnDuplicateSegmentLayer.REPLACE
    for layer in LAYERS:
        paths = [rrd_dir / layer_relpath(layer, base.stem) for base in base_rrds]
        uris = [p.resolve().as_uri() for p in paths if p.exists()]
        if not uris:
            print(f"  layer '{layer}': no files, skipping")
            continue
        dataset.register(uris, layer_name=layer, on_duplicate=on_duplicate).wait()
        print(f"  layer '{layer}': registered {len(uris)} file(s)")
        # A segment registered without its odom layer opens on an empty 3D view — the default
        # blueprint targets the `odom` frame, which only that layer defines. Say which episodes
        # are short rather than let the viewer report it as an unknown frame.
        for missing in (p for p in paths if not p.exists()):
            print(f"    missing: {missing.name} — this episode registers without its '{layer}' layer")

    if blueprint.exists():
        dataset.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
        print(f"  default blueprint: {blueprint}")
    else:
        print(f"  no blueprint at {blueprint} (run `pixi run -e hiw blueprint`)")
    return dataset


def main() -> None:
    """Register every converted episode into the local catalog as one dataset."""
    parser = argparse.ArgumentParser(description="Register HIW-500 RRDs into the local Rerun catalog.")
    parser.add_argument(
        "--rrd-dir",
        type=Path,
        default=dataset_rrd_dir(DATASET_NAME),
        help="Root of the per-layer recording directories (`<layer>/<recording_id>.rrd`).",
    )
    parser.add_argument(
        "--catalog-url", default=DEFAULT_CATALOG_URL, help="gRPC URL of the local catalog (`pixi run serve`)."
    )
    parser.add_argument(
        "--dataset-name", default=DATASET_NAME, help=f"Dataset to (re)create (default: {DATASET_NAME})."
    )
    parser.add_argument(
        "--blueprint", type=Path, default=BLUEPRINT_PATH, help="Blueprint (.rbl) to install as the dataset default."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete an existing dataset first and rebuild it from scratch.",
    )
    args = parser.parse_args()

    dataset = register_episodes(
        args.catalog_url, args.dataset_name, args.rrd_dir, args.blueprint, recreate=args.recreate
    )
    segments = dataset.segment_ids()
    print(f"Registered dataset '{dataset.name}' on {args.catalog_url} with {len(segments)} segment(s):")
    for segment in segments:
        print(f"  - {segment}")


if __name__ == "__main__":
    main()
