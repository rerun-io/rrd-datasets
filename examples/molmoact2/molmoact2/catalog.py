"""
Register the converted MolmoAct2 RRDs into a Rerun catalog as one dataset.

Each episode is one segment, keyed by its recording id, and its `.rrd` files attach as the layers
named in `LAYERS`. Each arm's model registers as an asset (`urdf-model-left`, `urdf-model-right`),
which the viewer loads with every segment, and the default blueprint is installed on the dataset.

Run:  pixi run serve                  # start the catalog, leave it running
      pixi run -e molmo register      # register every converted episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from molmoact2.base_layer import RRD_ROOT
from molmoact2.blueprint import BLUEPRINT_PATH
from molmoact2.layers import LAYERS
from molmoact2.urdf_layer import ARMS, model_recording_id, model_rrd_path
from rrd_datasets_common.paths import layer_relpath

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"
DATASET_NAME = "molmoact2"


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
    """Register each episode's layers as one segment, attach the arm model assets, and set the blueprint."""
    ids = recording_ids(rrd_dir)
    if not ids:
        raise FileNotFoundError(f"No base *.rrd files in {rrd_dir} (run `pixi run -e molmo convert-base`)")

    client = CatalogClient(catalog_url)
    if recreate and dataset_name in client.dataset_names():
        client.get_dataset(dataset_name).delete()
    dataset = client.create_dataset(dataset_name, exist_ok=True)

    for layer in LAYERS:
        paths = [rrd_dir / layer_relpath(layer, recording_id) for recording_id in ids]
        uris = [path.resolve().as_uri() for path in paths if path.exists()]
        if not uris:
            print(f"  layer '{layer}': no files, skipping")
            continue
        # REPLACE lets a second run update the layers already registered instead of failing on them.
        dataset.register(uris, layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        print(f"  layer '{layer}': registered {len(uris)} file(s)")
        for missing in (path for path in paths if not path.exists()):
            print(f"    missing: {missing.name} — this episode registers without its '{layer}' layer")

    for arm in ARMS:
        model = model_rrd_path(rrd_dir, arm)
        if model.exists():
            dataset.register_asset(model.resolve().as_uri())
            print(f"  asset '{model_recording_id(arm)}': {model}")
        else:
            print(
                f"  no model rrd at {model} — the {arm} arm shows without meshes (run `pixi run -e molmo convert-urdf`)"
            )

    if blueprint.exists():
        dataset.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
        print(f"  default blueprint: {blueprint}")
    else:
        print(f"  no blueprint at {blueprint} (run `pixi run -e molmo blueprint`)")
    return dataset


def main() -> None:
    """Register every converted episode into the catalog as one dataset."""
    parser = argparse.ArgumentParser(description="Register MolmoAct2 RRDs into a Rerun catalog.")
    parser.add_argument(
        "--rrd-dir", type=Path, default=RRD_ROOT, help="Root of the per-layer folders (`<layer>/<recording_id>.rrd`)."
    )
    parser.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL, help="Catalog gRPC URL (`pixi run serve`).")
    parser.add_argument("--dataset-name", default=DATASET_NAME, help="Dataset to create or update.")
    parser.add_argument(
        "--blueprint", type=Path, default=BLUEPRINT_PATH, help="Blueprint (.rbl) installed as the dataset default."
    )
    parser.add_argument("--recreate", action="store_true", help="Delete an existing dataset first.")
    args = parser.parse_args()

    dataset = register_episodes(
        args.catalog_url, args.dataset_name, args.rrd_dir, args.blueprint, recreate=args.recreate
    )
    print(f"Registered dataset '{dataset.name}' on {args.catalog_url} with {len(dataset.segment_ids())} segment(s)")


if __name__ == "__main__":
    main()
