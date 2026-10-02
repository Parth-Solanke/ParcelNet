#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.logging import logger
from backend.app.services.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 1: Ingestion and Georeferencing QC CLI"
    )
    parser.add_argument("--ori", type=Path, required=True, help="Path to input Orthorectified Image (GeoTIFF)")
    parser.add_argument("--dsm", type=Path, default=None, help="Path to input Digital Surface Model (GeoTIFF)")
    parser.add_argument("--dtm", type=Path, default=None, help="Path to input Digital Terrain Model (GeoTIFF)")
    parser.add_argument("--vector", type=Path, default=None, help="Path to optional existing vector parcel layer")
    parser.add_argument("--target-epsg", type=int, default=32643, help="Target EPSG CRS (default: 32643 UTM 43N)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")

    args = parser.parse_args()

    logger.info("Executing CadastraAI Ingestion & QC CLI...")
    pipeline = IngestionPipeline(target_crs_epsg=args.target_epsg, output_dir=args.output_dir)
    try:
        results = pipeline.process(
            ori_path=args.ori,
            dsm_path=args.dsm,
            dtm_path=args.dtm,
            vector_path=args.vector,
        )
        logger.info("Ingestion completed successfully!")
        print(json.dumps(results["qc_report"], indent=2))
        sys.exit(0 if results["qc_report"]["is_valid"] else 1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
