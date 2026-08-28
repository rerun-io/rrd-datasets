"""
Register the per-demo LIBERO RRDs into a local Rerun catalog as one dataset.

Each demo is one *segment*, keyed by the `recording_id` its RRDs share (`<suite>/<task>__<demo>`);
the RRDs attach as the `base`, `properties`, `urdf` and `cameras` *layers* of that segment, and the
default blueprint is installed on the dataset. Layer names are assigned at registration, not when
an RRD is written, which is why `rerun server -d rrds/libero` cannot load these files: every one
would land under the default layer `base`.

    pixi run serve               # start the in-memory Rerun catalog (leave running)
    pixi run -e libero register  # register every demo's layers into a dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from libero.blueprint import BLUEPRINT_PATH
from libero.layers import LAYERS
from rrd_datasets_common.paths import dataset_rrd_dir, layer_relpath

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"
DATASET_NAME = "libero"


def demo_ids(rrd_dir: Path) -> list[str]:
    """The recording ids under `rrd_dir`, read off the base layer's `<suite>/<task>__<demo>.rrd` files."""
    return sorted(f"{path.parent.name}/{path.stem}" for path in (rrd_dir / "base").glob("*/*.rrd"))


def register_demos(
    catalog_url: str,
    dataset_name: str,
    rrd_dir: Path,
    blueprint: Path,
    *,
    recreate: bool = False,
) -> DatasetEntry:
    """
    Register each demo's layers as one segment of the dataset and set the blueprint.

    The dataset is created if missing, and layers already there are replaced. With `recreate`, an
    existing dataset of the same name is deleted first and rebuilt from scratch.
    """
    ids = demo_ids(rrd_dir)
    if not ids:
        raise FileNotFoundError(f"No base */*.rrd files found in {rrd_dir}")

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
            print(f"    missing: {missing.relative_to(rrd_dir)} — this demo registers without its '{layer}' layer")

    if blueprint.exists():
        dataset.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
        print(f"  default blueprint: {blueprint}")
    else:
        print(f"  no blueprint at {blueprint} (run `pixi run -e libero blueprint`)")
    return dataset


def main() -> None:
    """Register every converted demo into the local catalog as one dataset."""
    parser = argparse.ArgumentParser(description="Register LIBERO RRDs into the local Rerun catalog.")
    parser.add_argument(
        "--rrd-dir",
        type=Path,
        default=dataset_rrd_dir(DATASET_NAME),
        help="Root of the per-layer recording directories (`<layer>/<suite>/<task>__<demo>.rrd`).",
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

    dataset = register_demos(args.catalog_url, args.dataset_name, args.rrd_dir, args.blueprint, recreate=args.recreate)
    segments = dataset.segment_ids()
    print(f"Registered dataset '{dataset.name}' on {args.catalog_url} with {len(segments)} segment(s):")
    for segment in segments:
        print(f"  - {segment}")


if __name__ == "__main__":
    main()
