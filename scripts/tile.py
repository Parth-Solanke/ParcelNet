#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio
from backend.app.core.logging import logger
from backend.app.services.tiling.window_tiler import WindowTiler
from backend.app.services.tiling.stitcher import TileStitcher


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 3: Sliding-Window Tiling and Overlap Blending Engine CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: tile
    tile_parser = subparsers.add_parser("tile", help="Generate sliding-window tile index or chips")
    tile_parser.add_argument("--raster", type=Path, required=True, help="Input GeoTIFF raster")
    tile_parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels (default: 1024)")
    tile_parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio (default: 0.25)")
    tile_parser.add_argument("--out-index", type=Path, default=Path("data/tiles/tile_index.geojson"), help="Output GeoJSON index")
    tile_parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory to save chip GeoTIFFs")

    # Sub-command: stitch
    stitch_parser = subparsers.add_parser("stitch", help="Tile and stitch back with overlap blending")
    stitch_parser.add_argument("--raster", type=Path, required=True, help="Input raster to test tile-stitch")
    stitch_parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels")
    stitch_parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio")
    stitch_parser.add_argument("--output", type=Path, required=True, help="Output reconstructed GeoTIFF")

    args = parser.parse_args()

    if args.command == "tile":
        logger.info(f"Computing tile windows for: {args.raster}")
        tiler = WindowTiler(tile_size_px=args.tile_size, overlap_ratio=args.overlap)
        index_path = tiler.export_tile_index_geojson(args.raster, args.out_index)
        logger.info(f"Tile index successfully created at: {index_path}")

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(args.raster) as src:
                for tile_data, win, meta in tiler.iter_tiles(args.raster):
                    chip_path = args.out_dir / f"tile_{meta.tile_id:04d}.tif"
                    profile = src.profile.copy()
                    profile.update({
                        "width": meta.width,
                        "height": meta.height,
                        "transform": meta.transform,
                    })
                    with rasterio.open(chip_path, "w", **profile) as dst:
                        dst.write(tile_data)
            logger.info(f"All tile chips saved to: {args.out_dir}")

    elif args.command == "stitch":
        logger.info(f"Running tile-and-stitch test on: {args.raster}")
        tiler = WindowTiler(tile_size_px=args.tile_size, overlap_ratio=args.overlap)
        with rasterio.open(args.raster) as src:
            stitcher = TileStitcher(
                full_width=src.width,
                full_height=src.height,
                channels=src.count,
                transform=src.transform,
                crs=src.crs,
                dtype=str(src.dtypes[0]),
            )
            for tile_data, win, meta in tiler.iter_tiles(args.raster):
                stitcher.add_tile(tile_data.astype(float), win)

            out_path = stitcher.save_to_geotiff(args.output)
            logger.info(f"Stitched raster successfully saved to: {out_path}")


if __name__ == "__main__":
    main()
