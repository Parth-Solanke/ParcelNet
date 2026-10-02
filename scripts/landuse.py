#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from backend.app.core.logging import logger
from backend.app.services.landuse.classifier import ObjectBasedLandUseRefiner


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 7: Object-Based Land-Use Classification CLI"
    )
    parser.add_argument("--parcels", type=Path, required=True, help="Input parcels vector layer (GeoJSON)")
    parser.add_argument("--landuse-raster", type=Path, required=True, help="Land-use probability or class GeoTIFF")
    parser.add_argument("--ndsm", type=Path, default=None, help="Optional nDSM raster for physical elevation rules")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/vectors/parcels_landuse.geojson"),
        help="Path to save classified parcels",
    )

    args = parser.parse_args()

    logger.info("Executing CadastraAI Land-Use Classification CLI...")
    parcels_gdf = gpd.read_file(args.parcels)
    refiner = ObjectBasedLandUseRefiner()

    try:
        classified_gdf = refiner.refine_parcels(
            parcels_gdf=parcels_gdf,
            landuse_raster_path=args.landuse_raster,
            ndsm_raster_path=args.ndsm,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        classified_gdf.to_file(args.output, driver="GeoJSON")
        logger.info(f"Classified parcels saved to: {args.output}")
    except Exception as e:
        logger.error(f"Land-use classification failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
