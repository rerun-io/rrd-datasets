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

from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from hiw_500.blueprint import BLUEPRINT_PATH
from hiw_500.layers import LAYERS
from rrd_datasets_common.paths import dataset_rrd_dir, layer_relpath

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"


def _base_rrds(rrd_dir: Path) -> list[Path]:
    """One base recording per episode: the `base/<recording_id>.rrd` files under `rrd_dir`."""
    return sorted((rrd_dir / "base").glob("*.rrd"))


def register_episodes(
    catalog_url: str,
    dataset_name: str,
    rrd_dir: Path,
    blueprint: Path,
    *,
    recreate: bool = True,
) -> DatasetEntry:
    """Create (or replace) the dataset, register each episode's layers, and set the blueprint."""
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


@dataclass
class CatalogConfig:
    """Configuration for registering the HIW-500 RRDs into a local catalog."""

    rrd_dir: Path = dataset_rrd_dir("hiw-500")
    """Root of the per-layer recording directories (`<layer>/<recording_id>.rrd`)."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun catalog (`rerun server`)."""
    dataset_name: str = "hiw_500"
    """Catalog dataset name to (re)create."""
    blueprint: Path = BLUEPRINT_PATH
    """Default blueprint (`.rbl`) to install on the dataset, if present."""
    recreate: bool = True
    """Delete and recreate the dataset before registering. Pass `--no-recreate` to
    re-register onto the existing dataset (REPLACE per layer)."""


def main(cfg: CatalogConfig) -> None:
    """Register every converted episode into the local catalog as one dataset."""
    dataset = register_episodes(cfg.catalog_url, cfg.dataset_name, cfg.rrd_dir, cfg.blueprint, recreate=cfg.recreate)
    segments = dataset.segment_ids()
    print(f"Registered dataset '{dataset.name}' on {cfg.catalog_url} with {len(segments)} segment(s):")
    for segment in segments:
        print(f"  - {segment}")


if __name__ == "__main__":
    main(tyro.cli(CatalogConfig, description="Register HIW-500 RRDs into the local Rerun catalog"))
