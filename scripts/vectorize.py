#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.logging import logger
from backend.app.services.vectorization.building_vectorizer import BuildingVectorizer
from backend.app.services.vectorization.road_vectorizer import RoadVectorizer


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 6: Feature Vectorization and Regularization CLI"
    )
    parser.add_argument("--buildings-prob", type=Path, default=None, help="Building probability GeoTIFF")
    parser.add_argument("--roads-prob", type=Path, default=None, help="Road probability GeoTIFF")
    parser.add_argument("--ndsm", type=Path, default=None, help="nDSM GeoTIFF for building height extraction")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/vectors"),
        help="Directory to save output vector layers",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Executing CadastraAI Feature Vectorization CLI...")

    if args.buildings_prob and args.buildings_prob.exists():
        b_vec = BuildingVectorizer()
        bldgs_gdf = b_vec.vectorize(args.buildings_prob, ndsm_path=args.ndsm)
        bldgs_out = args.output_dir / "buildings.geojson"
        bldgs_gdf.to_file(bldgs_out, driver="GeoJSON")
        logger.info(f"Saved {len(bldgs_gdf)} building footprints -> {bldgs_out}")

    if args.roads_prob and args.roads_prob.exists():
        r_vec = RoadVectorizer()
        c_lines_gdf, polys_gdf, _ = r_vec.vectorize(args.roads_prob)
        c_out = args.output_dir / "roads_centerlines.geojson"
        p_out = args.output_dir / "roads_polygons.geojson"
        c_lines_gdf.to_file(c_out, driver="GeoJSON")
        polys_gdf.to_file(p_out, driver="GeoJSON")
        logger.info(f"Saved {len(c_lines_gdf)} road centerlines -> {c_out}")
        logger.info(f"Saved {len(polys_gdf)} road polygons -> {p_out}")


if __name__ == "__main__":
    main()
