#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.logging import logger
from backend.app.services.preprocessing.stacker import FeatureStackBuilder


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 2: Pre-processing and Feature Layer Stack Builder CLI"
    )
    parser.add_argument("--ori", type=Path, required=True, help="Path to input Orthorectified Image (GeoTIFF)")
    parser.add_argument("--dsm", type=Path, default=None, help="Path to aligned Digital Surface Model (GeoTIFF)")
    parser.add_argument("--dtm", type=Path, default=None, help="Path to aligned Digital Terrain Model (GeoTIFF)")
    parser.add_argument(
        "--channels",
        type=str,
        default="rgb+ndsm",
        choices=["rgb", "rgb+ndsm", "rgb+ndsm+hillshade"],
        help="Feature stack channel combination preset",
    )
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE radiometric enhancement")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stack_4ch.tif"),
        help="Path for the output feature stack GeoTIFF",
    )

    args = parser.parse_args()

    logger.info("Executing CadastraAI Preprocessing CLI...")
    builder = FeatureStackBuilder(channel_preset=args.channels, use_clahe=args.clahe)
    try:
        output_path = builder.build_stack(
            ori_path=args.ori,
            dsm_path=args.dsm,
            dtm_path=args.dtm,
            output_stack_path=args.output,
        )
        logger.info(f"Preprocessing completed! Feature stack saved to: {output_path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
