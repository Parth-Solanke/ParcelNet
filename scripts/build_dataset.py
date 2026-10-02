#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from backend.app.core.logging import logger
from ml.datasets.dataset_builder import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 4: Training Dataset Builder and Label Rasterizer CLI"
    )
    parser.add_argument("--stack", type=Path, required=True, help="Path to 4-channel input stack GeoTIFF")
    parser.add_argument("--parcels", type=Path, default=None, help="Path to parcel boundaries/polygons vector layer")
    parser.add_argument("--buildings", type=Path, default=None, help="Path to buildings vector layer")
    parser.add_argument("--roads", type=Path, default=None, help="Path to roads vector layer")
    parser.add_argument("--landuse", type=Path, default=None, help="Path to landuse vector layer")
    parser.add_argument("--chip-size", type=int, default=512, help="Chip size in pixels (default: 512)")
    parser.add_argument("--stride", type=int, default=384, help="Sliding window stride (default: 384)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tiles/dataset"),
        help="Directory to save dataset chips and manifest",
    )

    args = parser.parse_args()

    logger.info("Executing CadastraAI Dataset Builder CLI...")
    builder = DatasetBuilder(chip_size_px=args.chip_size, stride_px=args.stride)

    parcels_gdf = gpd.read_file(args.parcels) if args.parcels and args.parcels.exists() else None
    buildings_gdf = gpd.read_file(args.buildings) if args.buildings and args.buildings.exists() else None
    roads_gdf = gpd.read_file(args.roads) if args.roads and args.roads.exists() else None
    landuse_gdf = gpd.read_file(args.landuse) if args.landuse and args.landuse.exists() else None

    try:
        manifest_path = builder.build_dataset(
            stack_path=args.stack,
            output_dir=args.output_dir,
            parcels_gdf=parcels_gdf,
            buildings_gdf=buildings_gdf,
            roads_gdf=roads_gdf,
            landuse_gdf=landuse_gdf,
        )
        logger.info(f"Dataset successfully created! Manifest at: {manifest_path}")
    except Exception as e:
        logger.error(f"Dataset build failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
