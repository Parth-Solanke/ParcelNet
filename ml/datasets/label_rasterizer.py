from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation, distance_transform_edt
from shapely.geometry import GeometryCollection, MultiLineString, MultiPolygon, Polygon


class LabelRasterizer:
    """Rasterizes vector annotations (buildings, roads, parcels, landuse) into multi-task masks."""

    LANDUSE_CLASSES = {
        "background": 0,
        "residential": 1,
        "commercial": 2,
        "institutional": 3,
        "open_land": 4,
        "vegetation": 5,
        "water": 6,
        "road": 7,
        "built_up_other": 8,
    }

    def __init__(self, boundary_dilation_px: int = 2, max_dist_px: float = 30.0):
        self.boundary_dilation_px = boundary_dilation_px
        self.max_dist_px = max_dist_px

    def rasterize_polygons(
        self,
        gdf: gpd.GeoDataFrame,
        shape: Tuple[int, int],
        transform: Affine,
        all_touched: bool = False,
    ) -> np.ndarray:
        """Rasterizes polygon geometry interiors into binary uint8 mask (0 or 1)."""
        height, width = shape
        if gdf is None or len(gdf) == 0:
            return np.zeros((height, width), dtype=np.uint8)

        # Filter out empty or non-polygon geometries
        valid_geoms = [
            (geom, 1)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty and geom.is_valid
        ]
        if not valid_geoms:
            return np.zeros((height, width), dtype=np.uint8)

        mask = rasterize(
            shapes=valid_geoms,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype=np.uint8,
        )
        return mask

    def rasterize_boundaries_and_distance(
        self,
        gdf: gpd.GeoDataFrame,
        shape: Tuple[int, int],
        transform: Affine,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rasterizes polygon boundaries into:
        1. Dilated binary boundary mask (2-3px wide)
        2. Boundary distance map [0.0, 1.0] where 1.0 is on the boundary and decays with distance.
        """
        height, width = shape
        if gdf is None or len(gdf) == 0:
            return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)

        boundary_lines = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty or not geom.is_valid:
                continue
            if isinstance(geom, (Polygon, MultiPolygon)):
                boundary = geom.boundary
                if not boundary.is_empty:
                    boundary_lines.append((boundary, 1))

        if not boundary_lines:
            return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)

        # 1-pixel raw boundary lines
        raw_boundary = rasterize(
            shapes=boundary_lines,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )

        # Dilate by 2-3 pixels to handle thin line imbalance
        if self.boundary_dilation_px > 0:
            structure = np.ones((self.boundary_dilation_px * 2 + 1, self.boundary_dilation_px * 2 + 1), dtype=bool)
            dilated_boundary = binary_dilation(raw_boundary > 0, structure=structure).astype(np.uint8)
        else:
            dilated_boundary = raw_boundary

        # Euclidean distance transform from boundary
        # edt computes distance to nearest zero (background) pixel, so invert mask
        dist = distance_transform_edt(raw_boundary == 0)
        # Normalized distance target: 1.0 on boundary, decaying exponentially or linearly to 0
        norm_dist = np.clip(1.0 - (dist / self.max_dist_px), 0.0, 1.0).astype(np.float32)

        return dilated_boundary, norm_dist

    def rasterize_landuse(
        self,
        gdf: gpd.GeoDataFrame,
        shape: Tuple[int, int],
        transform: Affine,
        class_column: str = "class",
    ) -> np.ndarray:
        """Rasterizes landuse polygons into a multi-class index mask (0 to 8)."""
        height, width = shape
        if gdf is None or len(gdf) == 0 or class_column not in gdf.columns:
            return np.zeros((height, width), dtype=np.uint8)

        shapes_with_values = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.is_valid:
                continue
            class_name = str(row[class_column]).lower().strip()
            class_id = self.LANDUSE_CLASSES.get(class_name, 0)
            shapes_with_values.append((geom, class_id))

        if not shapes_with_values:
            return np.zeros((height, width), dtype=np.uint8)

        mask = rasterize(
            shapes=shapes_with_values,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.uint8,
        )
        return mask
