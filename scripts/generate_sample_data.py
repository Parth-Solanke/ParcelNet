#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.logging import logger
from tests.fixtures.synthetic_drone_data import generate_synthetic_drone_dataset


def main():
    target_dir = Path("data/raw")
    logger.info(f"Generating synthetic drone dataset in: {target_dir}")
    dataset = generate_synthetic_drone_dataset(
        output_dir=target_dir,
        width=500,
        height=500,
        gsd=0.05,
        epsg=32643,
    )
    logger.info("Sample dataset generated successfully:")
    for k, v in dataset.items():
        logger.info(f"  {k.upper()}: {v}")


if __name__ == "__main__":
    main()
