from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, label as scipy_label
from skimage.segmentation import watershed
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.validation import make_valid
from backend.app.core.logging import logger


class BuildingVectorizer:
    """Vectorizes building probability rasters with watershed instance separation and orthogonalization."""

    def __init__(
        self,
        prob_threshold: float = 0.50,
        min_area_sqm: float = 4.0,
        simplify_tolerance_ratio: float = 1.5,
        min_orthogonal_iou: float = 0.88,
    ):
        self.prob_threshold = prob_threshold
        self.min_area_sqm = min_area_sqm
        self.simplify_tolerance_ratio = simplify_tolerance_ratio
        self.min_orthogonal_iou = min_orthogonal_iou

    def separate_instances_watershed(
        self,
        binary_mask: np.ndarray,
        ndsm_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Separates touching buildings using distance transform and nDSM peak seeding."""
        dist = distance_transform_edt(binary_mask)
        if ndsm_data is not None:
            # Combine geometric distance with elevation peak evidence
            dist = dist * (1.0 + np.clip(ndsm_data / 10.0, 0.0, 2.0))

        # Local maxima detection as seeds
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(dist, kernel)
        peaks = (dist == dilated) & (dist > 2.0)

        markers, num_markers = scipy_label(peaks)
        if num_markers <= 1:
            labeled, _ = scipy_label(binary_mask)
            return labeled

        labels = watershed(-dist, markers, mask=binary_mask)
        return labels

    def orthogonalize_polygon(self, poly: Polygon) -> Polygon:
        """
        Dominant-angle regularisation:
        Finds the minimum rotated rectangle angle and snaps near-perpendicular edges
        while ensuring IoU >= min_orthogonal_iou with the original polygon.
        """
        if not poly.is_valid or poly.area < 1.0:
            return poly

        mrr = poly.minimum_rotated_rectangle
        if not isinstance(mrr, Polygon) or len(mrr.exterior.coords) < 4:
            return poly

        coords = list(mrr.exterior.coords)
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        dominant_angle = np.degrees(np.arctan2(dy, dx))

        # Rotate to align with principal axis
        rotated_poly = rotate(poly, -dominant_angle, origin="centroid")
        # Snap bbox or simplify along Manhattan grid
        envelope = rotated_poly.envelope
        # Re-rotate back
        regularized = rotate(envelope, dominant_angle, origin="centroid")

        # IoU check: keep regularized only if high fidelity with original shape
        if poly.intersects(regularized):
            inter = poly.intersection(regularized).area
            union = poly.union(regularized).area
            if union > 0 and (inter / union) >= self.min_orthogonal_iou:
                return regularized

        return poly

    def vectorize(
        self,
        prob_raster_path: Path,
        ndsm_path: Optional[Path] = None,
        separate_instances: bool = True,
    ) -> gpd.GeoDataFrame:
        """Turns building probability map into clean, orthogonalized vector polygons with median heights."""
        with rasterio.open(prob_raster_path) as src:
            prob_arr = src.read(1)
            transform = src.transform
            crs = src.crs
            gsd = (abs(transform[0]) + abs(transform[4])) / 2.0
            simplify_tol = self.simplify_tolerance_ratio * gsd

        ndsm_arr = None
        if ndsm_path and ndsm_path.exists():
            with rasterio.open(ndsm_path) as ndsm_src:
                ndsm_arr = ndsm_src.read(1)

        # 1. Threshold & morphological cleanup
        binary = (prob_arr >= self.prob_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 2. Instance separation
        if separate_instances:
            labeled = self.separate_instances_watershed(binary, ndsm_arr)
        else:
            labeled, _ = scipy_label(binary)

        # 3. Polygonise
        features: List[Dict[str, Any]] = []
        building_id = 1

        for geom_dict, label_val in shapes(labeled.astype(np.int32), mask=(labeled > 0), transform=transform):
            poly_geom = shape(geom_dict)
            if not poly_geom.is_valid:
                poly_geom = make_valid(poly_geom)

            # Extract Polygon from potential MultiPolygon
            polys = [poly_geom] if isinstance(poly_geom, Polygon) else [p for p in poly_geom.geoms if isinstance(p, Polygon)]

            for p in polys:
                if p.area < self.min_area_sqm:
                    continue

                # Douglas-Peucker simplification
                simplified = p.simplify(tolerance=simplify_tol, preserve_topology=True)
                if not simplified.is_valid or simplified.is_empty:
                    simplified = p

                # Orthogonalization
                ortho_poly = self.orthogonalize_polygon(simplified)

                # Sample confidence & height from nDSM inside bounding region
                # Use centroid or rasterio window sampling
                px, py = ~transform * (ortho_poly.centroid.x, ortho_poly.centroid.y)
                px, py = int(np.clip(px, 0, prob_arr.shape[1] - 1)), int(np.clip(py, 0, prob_arr.shape[0] - 1))
                conf = float(prob_arr[py, px])

                height_m = 0.0
                if ndsm_arr is not None:
                    height_m = float(ndsm_arr[py, px])

                features.append({
                    "id": f"bldg_{building_id:05d}",
                    "confidence": round(conf, 4),
                    "height_m": round(height_m, 2),
                    "area_sqm": round(float(ortho_poly.area), 2),
                    "geometry": ortho_poly,
                })
                building_id += 1

        if features:
            gdf = gpd.GeoDataFrame(features, crs=crs)
        else:
            gdf = gpd.GeoDataFrame(
                columns=["id", "confidence", "height_m", "area_sqm", "geometry"],
                geometry="geometry",
                crs=crs,
            )
        logger.info(f"Vectorized {len(gdf)} building footprints from {prob_raster_path.name}")
        return gdf
