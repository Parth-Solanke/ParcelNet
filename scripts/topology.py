#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from backend.app.core.logging import logger
from backend.app.services.topology.line_network import BoundaryLineNetworkBuilder
from backend.app.services.topology.polygonizer import FacePolygonizer


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 8: Topology Engine and Parcel Polygon Creation CLI"
    )
    parser.add_argument("--boundaries-prob", type=Path, required=True, help="Parcel boundary probability GeoTIFF")
    parser.add_argument("--roads", type=Path, default=None, help="Road polygons layer (GeoJSON)")
    parser.add_argument("--buildings", type=Path, default=None, help="Building footprints layer (GeoJSON)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/vectors/parcels.geojson"),
        help="Output path for parcel polygons (GeoJSON)",
    )

    args = parser.parse_args()

    logger.info("Executing CadastraAI Topology Engine CLI...")
    road_polys = gpd.read_file(args.roads) if args.roads and args.roads.exists() else None
    bldg_polys = gpd.read_file(args.buildings) if args.buildings and args.buildings.exists() else None

    line_builder = BoundaryLineNetworkBuilder()
    noded_lines = line_builder.build_network(
        boundaries_prob_path=args.boundaries_prob,
        road_polygons=road_polys,
        building_polygons=bldg_polys,
    )

    polygonizer = FacePolygonizer()
    parcels_gdf = polygonizer.polygonize_network(
        noded_lines=noded_lines,
        road_polygons=road_polys,
        building_polygons=bldg_polys,
        crs=road_polys.crs if road_polys is not None else None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parcels_gdf.to_file(args.output, driver="GeoJSON")
    logger.info(f"Parcel generation completed! Saved {len(parcels_gdf)} parcels -> {args.output}")


if __name__ == "__main__":
    main()
