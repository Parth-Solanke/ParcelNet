from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
from backend.app.core.logging import logger


class FacePolygonizer:
    """
    Polygonizes noded line networks into closed cadastral faces,
    filters road/exterior faces, and merges slivers into adjacent neighbours.
    """

    def __init__(
        self,
        min_area_sqm: float = 5.0,
        sliver_area_sqm: float = 15.0,
        sliver_compactness_threshold: float = 0.05,
    ):
        self.min_area_sqm = min_area_sqm
        self.sliver_area_sqm = sliver_area_sqm
        self.sliver_compactness = sliver_compactness_threshold

    @staticmethod
    def calculate_compactness(poly: Polygon) -> float:
        """Polsby-Popper compactness = 4 * pi * Area / Perimeter^2."""
        if poly.length <= 0:
            return 0.0
        return float((4.0 * np.pi * poly.area) / (poly.length**2))

    def merge_slivers(self, candidate_polys: List[Polygon]) -> List[Polygon]:
        """Merges sliver polygons into the adjacent neighbor sharing the longest border."""
        if len(candidate_polys) <= 1:
            return candidate_polys

        valid_faces = []
        slivers = []

        for p in candidate_polys:
            comp = self.calculate_compactness(p)
            if p.area < self.sliver_area_sqm or comp < self.sliver_compactness:
                slivers.append(p)
            else:
                valid_faces.append(p)

        if not valid_faces:
            return candidate_polys  # Return original if all were marked as slivers

        tree = STRtree(valid_faces)
        for sliv in slivers:
            neighbors_idx = tree.query(sliv)
            best_neighbor_idx = None
            max_shared_length = 0.0

            for n_idx in neighbors_idx:
                neighbor = valid_faces[n_idx]
                if sliv.touches(neighbor) or sliv.intersects(neighbor):
                    inter = sliv.intersection(neighbor)
                    shared_len = inter.length
                    if shared_len > max_shared_length:
                        max_shared_length = shared_len
                        best_neighbor_idx = n_idx

            if best_neighbor_idx is not None:
                # Merge sliver into best neighbor
                merged = unary_union([valid_faces[best_neighbor_idx], sliv])
                if isinstance(merged, Polygon) and merged.is_valid:
                    valid_faces[best_neighbor_idx] = merged
            else:
                # If no neighbor found, retain sliver if above absolute min area
                if sliv.area >= self.min_area_sqm:
                    valid_faces.append(sliv)

        return valid_faces

    def polygonize_network(
        self,
        noded_lines: List[Any],
        road_polygons: Optional[gpd.GeoDataFrame] = None,
        building_polygons: Optional[gpd.GeoDataFrame] = None,
        crs: Any = None,
    ) -> gpd.GeoDataFrame:
        """
        Polygonizes noded lines, filters road surfaces, merges slivers,
        and computes cadastral attributes.
        """
        raw_faces = list(polygonize(noded_lines))
        logger.info(f"Polygonized {len(raw_faces)} raw closed faces from network")

        # 1. Filter out road faces
        road_union = unary_union(road_polygons.geometry) if road_polygons is not None and len(road_polygons) > 0 else None

        filtered_faces = []
        for face in raw_faces:
            if face.area < self.min_area_sqm or not face.is_valid:
                continue

            # If face is inside or largely overlapping road surface (>50%), drop it
            if road_union is not None and face.intersects(road_union):
                overlap_area = face.intersection(road_union).area
                if (overlap_area / face.area) > 0.50:
                    continue

            filtered_faces.append(face)

        # 2. Merge slivers
        clean_faces = self.merge_slivers(filtered_faces)

        # 3. Compute parcel attributes
        parcels_data = []
        parcel_id = 1
        bldg_tree = STRtree(list(building_polygons.geometry)) if building_polygons is not None and len(building_polygons) > 0 else None

        for face in clean_faces:
            compactness = self.calculate_compactness(face)
            area = float(face.area)
            perimeter = float(face.length)

            # Contained buildings count
            bldg_count = 0
            if bldg_tree is not None:
                contained_idx = bldg_tree.query(face, predicate="contains")
                bldg_count = len(contained_idx)

            # Heuristic confidence score
            conf = min(0.95, max(0.50, compactness * 0.7 + (0.25 if bldg_count > 0 else 0.10)))
            needs_gt = bool(conf < 0.65)

            parcels_data.append({
                "id": f"pcl_{parcel_id:05d}",
                "area_sqm": round(area, 2),
                "perimeter_m": round(perimeter, 2),
                "compactness": round(compactness, 4),
                "buildings_count": bldg_count,
                "confidence": round(conf, 4),
                "source": "ai",
                "status": "auto",
                "needs_gt": needs_gt,
                "geometry": face,
            })
            parcel_id += 1

        if parcels_data:
            gdf = gpd.GeoDataFrame(parcels_data, crs=crs)
        else:
            gdf = gpd.GeoDataFrame(
                columns=["id", "area_sqm", "perimeter_m", "compactness", "buildings_count", "confidence", "source", "status", "needs_gt", "geometry"],
                geometry="geometry",
                crs=crs,
            )
        logger.info(f"Generated {len(gdf)} topologically clean parcel polygons")
        return gdf
