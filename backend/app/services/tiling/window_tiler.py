from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from backend.app.core.logging import logger


class TileMetadata:
    def __init__(
        self,
        tile_id: int,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        transform: Affine,
        bounds: Tuple[float, float, float, float],
    ):
        self.tile_id = tile_id
        self.col_off = col_off
        self.row_off = row_off
        self.width = width
        self.height = height
        self.transform = transform
        self.bounds = bounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "col_off": self.col_off,
            "row_off": self.row_off,
            "width": self.width,
            "height": self.height,
            "transform": list(self.transform)[:6],
            "bounds": list(self.bounds),
        }


class WindowTiler:
    """Sliding-window raster tiler with overlap blending support and nodata filtering."""

    def __init__(
        self,
        tile_size_px: int = 1024,
        overlap_ratio: float = 0.25,
        max_nodata_ratio: float = 0.70,
    ):
        self.tile_size = tile_size_px
        self.overlap_ratio = overlap_ratio
        self.stride = int(round(tile_size_px * (1.0 - overlap_ratio)))
        self.max_nodata_ratio = max_nodata_ratio

    def compute_tile_windows(
        self,
        img_width: int,
        img_height: int,
        transform: Affine,
    ) -> List[Tuple[Window, TileMetadata]]:
        """Calculates sliding windows and associated metadata covering the full image dimensions."""
        tiles: List[Tuple[Window, TileMetadata]] = []
        tile_id = 0

        # Iterate rows
        y_offsets: List[int] = list(range(0, max(1, img_height - self.tile_size + 1), self.stride))
        if y_offsets[-1] + self.tile_size < img_height:
            y_offsets.append(img_height - self.tile_size)
        elif not y_offsets:
            y_offsets = [0]

        # Iterate columns
        x_offsets: List[int] = list(range(0, max(1, img_width - self.tile_size + 1), self.stride))
        if x_offsets[-1] + self.tile_size < img_width:
            x_offsets.append(img_width - self.tile_size)
        elif not x_offsets:
            x_offsets = [0]

        # Deduplicate and sort
        y_offsets = sorted(list(set(y_offsets)))
        x_offsets = sorted(list(set(x_offsets)))

        for row_off in y_offsets:
            win_h = min(self.tile_size, img_height - row_off)
            for col_off in x_offsets:
                win_w = min(self.tile_size, img_width - col_off)
                win = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
                tile_transform = rasterio.windows.transform(win, transform)

                minx, maxy = tile_transform * (0, 0)
                maxx, miny = tile_transform * (win_w, win_h)
                bounds = (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

                meta = TileMetadata(
                    tile_id=tile_id,
                    col_off=col_off,
                    row_off=row_off,
                    width=win_w,
                    height=win_h,
                    transform=tile_transform,
                    bounds=bounds,
                )
                tiles.append((win, meta))
                tile_id += 1

        return tiles

    def iter_tiles(
        self,
        raster_path: Path,
    ) -> Generator[Tuple[np.ndarray, Window, TileMetadata], None, None]:
        """Generator yielding (tile_data, window, tile_metadata) for each tile in raster."""
        with rasterio.open(raster_path) as src:
            tiles = self.compute_tile_windows(src.width, src.height, src.transform)
            for win, meta in tiles:
                tile_data = src.read(window=win)
                yield tile_data, win, meta

    def export_tile_index_geojson(
        self,
        raster_path: Path,
        output_geojson_path: Path,
    ) -> Path:
        """Generates a GeoJSON feature collection indexing all tile bounding boxes."""
        output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(raster_path) as src:
            tiles = self.compute_tile_windows(src.width, src.height, src.transform)
            crs_code = src.crs.to_string() if src.crs else "EPSG:32643"

        features = []
        for _, meta in tiles:
            geom = box(*meta.bounds)
            feat = {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": meta.to_dict(),
            }
            features.append(feat)

        geojson_data = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs_code}},
            "features": features,
        }

        with open(output_geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        logger.info(f"Exported tile index GeoJSON ({len(features)} tiles) to {output_geojson_path}")
        return output_geojson_path
