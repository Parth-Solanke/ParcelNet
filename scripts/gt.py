#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from backend.app.core.logging import logger
from backend.app.services.gt_integration.importer import GTImporter
from backend.app.services.gt_integration.matcher import GTMatcher
from backend.app.services.gt_integration.verification_list import FieldVerificationExporter


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 10: Ground Truthing & Field Verification CLI"
    )
    parser.add_argument("--parcels", type=Path, required=True, help="Input parcels layer (GeoJSON)")
    parser.add_argument("--gt-points", type=Path, default=None, help="Input GT points (CSV or GeoJSON)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gt"),
        help="Directory to save verified layers and worklists",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Executing CadastraAI Ground Truthing & Verification CLI...")

    parcels_gdf = gpd.read_file(args.parcels)
    importer = GTImporter()
    matcher = GTMatcher()
    exporter = FieldVerificationExporter()

    if args.gt_points and args.gt_points.exists():
        if args.gt_points.suffix.lower() == ".csv":
            gt_gdf = importer.import_points_csv(args.gt_points)
        else:
            gt_gdf = importer.import_vector(args.gt_points)

        verified_parcels, metrics = matcher.match_points_to_parcels(parcels_gdf, gt_gdf)
        verified_path = args.output_dir / "parcels_gt_verified.geojson"
        verified_parcels.to_file(verified_path, driver="GeoJSON")
        logger.info(f"Saved GT-verified parcels -> {verified_path}")
        print(json.dumps(metrics, indent=2))
        parcels_gdf = verified_parcels

    # Export field verification worklist
    v_list = exporter.create_verification_list(parcels_gdf, output_dir=args.output_dir)
    logger.info(f"Field verification list generated with {len(v_list)} parcels.")


if __name__ == "__main__":
    main()
