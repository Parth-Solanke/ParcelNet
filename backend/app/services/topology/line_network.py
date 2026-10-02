from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from shapely import snap
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import linemerge, unary_union
from backend.app.core.logging import logger


class BoundaryLineNetworkBuilder:
    """Extracts boundary skeleton lines, connects endpoints, snaps nodes, and nodes lines into a clean network."""

    def __init__(
        self,
        prob_threshold: float = 0.35,
        min_spur_length_m: float = 2.0,
        gap_connect_tolerance_m: float = 2.5,
        snap_tolerance_m: float = 0.30,
    ):
        self.prob_threshold = prob_threshold
        self.min_spur_m = min_spur_length_m
        self.gap_tol_m = gap_connect_tolerance_m
        self.snap_tol_m = snap_tolerance_m

    def trace_skeleton(self, skeleton: np.ndarray, transform: Affine) -> List[LineString]:
        """Traces skeleton pixels into LineStrings."""
        height, width = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        lines: List[LineString] = []

        for r in range(height):
            for c in range(width):
                if skeleton[r, c] and not visited[r, c]:
                    coords = []
                    curr_r, curr_c = r, c
                    while True:
                        visited[curr_r, curr_c] = True
                        gx, gy = transform * (curr_c + 0.5, curr_r + 0.5)
                        coords.append((gx, gy))

                        next_pixel = None
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < height and 0 <= nc < width and skeleton[nr, nc] and not visited[nr, nc]:
                                    next_pixel = (nr, nc)
                                    break
                            if next_pixel:
                                break

                        if next_pixel:
                            curr_r, curr_c = next_pixel
                        else:
                            break

                    if len(coords) >= 2:
                        lines.append(LineString(coords))

        return lines

    def bridge_near_endpoints(self, lines: List[LineString]) -> List[LineString]:
        """Bridges gaps between nearby endpoints within gap_connect_tolerance_m."""
        if not lines:
            return []

        endpoints = []
        for idx, line in enumerate(lines):
            c = list(line.coords)
            endpoints.append((c[0], idx, 0))
            endpoints.append((c[-1], idx, -1))

        bridges = []
        n_pts = len(endpoints)
        for i in range(n_pts):
            pt_i, l_i, _ = endpoints[i]
            for j in range(i + 1, n_pts):
                pt_j, l_j, _ = endpoints[j]
                if l_i == l_j:
                    continue
                dist = np.hypot(pt_i[0] - pt_j[0], pt_i[1] - pt_j[1])
                if 0.05 < dist <= self.gap_tol_m:
                    bridges.append(LineString([pt_i, pt_j]))

        return lines + bridges

    def build_network(
        self,
        boundaries_prob_path: Path,
        road_polygons: Optional[gpd.GeoDataFrame] = None,
        building_polygons: Optional[gpd.GeoDataFrame] = None,
    ) -> List[LineString]:
        """
        Extracts boundary network and fuses constraint lines from road edges and building perimeters.
        """
        with rasterio.open(boundaries_prob_path) as src:
            prob = src.read(1)
            transform = src.transform

        # 1. Threshold & Skeletonize
        binary = (prob >= self.prob_threshold).astype(np.uint8)
        skel = skeletonize(binary > 0)

        # 2. Trace Lines
        raw_lines = self.trace_skeleton(skel, transform)
        # Filter spurs
        filtered_lines = [line for line in raw_lines if line.length >= self.min_spur_m]

        # 3. Bridge small gaps
        bridged_lines = self.bridge_near_endpoints(filtered_lines)

        # 4. Integrate Constraint Lines
        all_lines = list(bridged_lines)
        if road_polygons is not None and len(road_polygons) > 0:
            for geom in road_polygons.geometry:
                if geom is not None and not geom.is_empty and geom.is_valid:
                    boundary = geom.boundary
                    if isinstance(boundary, LineString):
                        all_lines.append(boundary)
                    elif isinstance(boundary, MultiLineString):
                        all_lines.extend(list(boundary.geoms))

        if not all_lines:
            return []

        # 5. Snap and Node Network using unary_union
        union_geom = unary_union(all_lines)
        noded_lines = []
        if isinstance(union_geom, LineString):
            noded_lines.append(union_geom)
        elif isinstance(union_geom, MultiLineString):
            noded_lines.extend(list(union_geom.geoms))

        logger.info(f"Built noded boundary line network: {len(noded_lines)} segments")
        return noded_lines
