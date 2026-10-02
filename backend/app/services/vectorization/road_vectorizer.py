from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import linemerge, unary_union
from backend.app.core.logging import logger


class RoadVectorizer:
    """Vectorizes road probability rasters into clean centerlines, polygons, and network graphs."""

    def __init__(
        self,
        prob_threshold: float = 0.45,
        min_spur_length_m: float = 3.0,
        gap_connect_tolerance_m: float = 5.0,
        default_width_m: float = 3.5,
    ):
        self.prob_threshold = prob_threshold
        self.min_spur_m = min_spur_length_m
        self.gap_tol_m = gap_connect_tolerance_m
        self.default_width_m = default_width_m

    def trace_skeleton_lines(
        self,
        skeleton: np.ndarray,
        dist_map: np.ndarray,
        transform: Affine,
        gsd: float,
    ) -> List[Dict[str, Any]]:
        """Traces skeleton pixels into LineString segments with estimated average road widths."""
        height, width = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        segments: List[Dict[str, Any]] = []

        # Find 8-neighborhood connections
        for r in range(height):
            for c in range(width):
                if skeleton[r, c] and not visited[r, c]:
                    # Trace a simple polyline path
                    path_coords = []
                    widths = []
                    curr_r, curr_c = r, c

                    while True:
                        visited[curr_r, curr_c] = True
                        gx, gy = transform * (curr_c + 0.5, curr_r + 0.5)
                        path_coords.append((gx, gy))
                        widths.append(float(dist_map[curr_r, curr_c] * 2.0 * gsd))

                        # Check unvisited neighbors
                        found_next = False
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < height and 0 <= nc < width:
                                    if skeleton[nr, nc] and not visited[nr, nc]:
                                        curr_r, curr_c = nr, nc
                                        found_next = True
                                        break
                            if found_next:
                                break
                        if not found_next:
                            break

                    if len(path_coords) >= 2:
                        line = LineString(path_coords)
                        avg_width = float(np.mean(widths)) if widths else self.default_width_m
                        segments.append({
                            "geometry": line,
                            "width_m": max(avg_width, 1.5),
                            "length_m": float(line.length),
                        })

        return segments

    def build_network_graph(self, lines: List[LineString]) -> nx.Graph:
        """Constructs a networkx spatial graph from centerlines to identify junctions and dead-ends."""
        G = nx.Graph()
        for idx, line in enumerate(lines):
            coords = list(line.coords)
            start_node = (round(coords[0][0], 2), round(coords[0][1], 2))
            end_node = (round(coords[-1][0], 2), round(coords[-1][1], 2))
            G.add_node(start_node, pos=start_node)
            G.add_node(end_node, pos=end_node)
            G.add_edge(start_node, end_node, weight=line.length, line_id=idx)
        return G

    def vectorize(
        self,
        prob_raster_path: Path,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.Graph]:
        """
        Extracts centerlines and buffered road polygons.
        Returns: (centerlines_gdf, polygons_gdf, road_graph).
        """
        with rasterio.open(prob_raster_path) as src:
            prob_arr = src.read(1)
            transform = src.transform
            crs = src.crs
            gsd = (abs(transform[0]) + abs(transform[4])) / 2.0

        # Threshold and close small gaps
        binary = (prob_arr >= self.prob_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Distance transform for road width
        dist_map = distance_transform_edt(binary)

        # Skeletonization
        skel = skeletonize(binary > 0)

        # Trace skeleton into linestring segments
        raw_segments = self.trace_skeleton_lines(skel, dist_map, transform, gsd)

        # Filter out tiny spurs
        cleaned_lines = []
        centerlines_data = []
        polygons_data = []
        road_id = 1

        for seg in raw_segments:
            line_geom = seg["geometry"]
            if line_geom.length < self.min_spur_m:
                continue

            cleaned_lines.append(line_geom)
            width = seg["width_m"]

            # Buffer centerline to create road surface polygon
            poly_geom = line_geom.buffer(width / 2.0, cap_style="flat", join_style="round")

            centerlines_data.append({
                "id": f"road_{road_id:04d}",
                "width_m": round(width, 2),
                "length_m": round(float(line_geom.length), 2),
                "type": "local",
                "confidence": 0.85,
                "geometry": line_geom,
            })

            polygons_data.append({
                "id": f"road_poly_{road_id:04d}",
                "road_id": f"road_{road_id:04d}",
                "width_m": round(width, 2),
                "geometry": poly_geom,
            })
            road_id += 1

        # Network Graph
        road_graph = self.build_network_graph(cleaned_lines)

        if centerlines_data:
            centerlines_gdf = gpd.GeoDataFrame(centerlines_data, crs=crs)
        else:
            centerlines_gdf = gpd.GeoDataFrame(
                columns=["id", "width_m", "length_m", "type", "confidence", "geometry"],
                geometry="geometry",
                crs=crs,
            )

        if polygons_data:
            polygons_gdf = gpd.GeoDataFrame(polygons_data, crs=crs)
        else:
            polygons_gdf = gpd.GeoDataFrame(
                columns=["id", "road_id", "width_m", "geometry"],
                geometry="geometry",
                crs=crs,
            )

        logger.info(
            f"Extracted {len(centerlines_gdf)} road centerlines and {road_graph.number_of_nodes()} "
            f"graph junctions from {prob_raster_path.name}"
        )
        return centerlines_gdf, polygons_gdf, road_graph
