#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from backend.app.core.logging import logger
from backend.app.services.validation.auto_fixer import AutoFixEngine
from backend.app.services.validation.validator import CadastraValidator


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 9: Topology and Geometry Validation CLI"
    )
    parser.add_argument("--parcels", type=Path, required=True, help="Input parcels layer (GeoJSON)")
    parser.add_argument("--fix", action="store_true", help="Automatically apply geometry and overlap healing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/validation"),
        help="Directory to save validation reports and issues",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Executing CadastraAI Geometry & Topology Validation CLI...")

    parcels_gdf = gpd.read_file(args.parcels)
    validator = CadastraValidator()

    # Pre-validation
    report = validator.validate(parcels_gdf)
    summary = report["summary"]
    issues_gdf = report["issues_gdf"]

    report_path = args.output_dir / "validation_summary.json"
    issues_path = args.output_dir / "validation_issues.geojson"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    issues_gdf.to_file(issues_path, driver="GeoJSON")

    logger.info(
        f"Validation complete: {summary['total_issues']} issues found. "
        f"Clean parcels: {summary['clean_percentage']}%"
    )

    if args.fix:
        logger.info("Applying automated geometry healing...")
        fixer = AutoFixEngine()
        healed_gdf, audit = fixer.auto_fix_all(parcels_gdf)
        healed_path = args.output_dir / "parcels_healed.geojson"
        healed_gdf.to_file(healed_path, driver="GeoJSON")

        # Post-healing validation
        post_report = validator.validate(healed_gdf)
        post_summary = post_report["summary"]
        post_path = args.output_dir / "post_fix_validation_summary.json"
        with open(post_path, "w", encoding="utf-8") as f:
            json.dump(post_summary, f, indent=2)

        logger.info(
            f"Healing completed! Healed parcels saved to: {healed_path}. "
            f"New clean percentage: {post_summary['clean_percentage']}%"
        )


if __name__ == "__main__":
    main()
