from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree
from backend.app.core.logging import logger


class GTMatcher:
    """Matches ground truth surveys with AI parcel geometries and computes agreement metrics."""

    def __init__(self, max_boundary_offset_tolerance_m: float = 0.50):
        self.max_offset_tol = max_boundary_offset_tolerance_m

    def match_points_to_parcels(
        self,
        parcels_gdf: gpd.GeoDataFrame,
        gt_points_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Calculates distance from GT corner/boundary points to nearest AI parcel boundary.
        Updates status to 'verified' if average offset is within tolerance.
        """
        updated_parcels = parcels_gdf.copy()
        if "status" not in updated_parcels.columns:
            updated_parcels["status"] = "auto"
        if "gt_offset_m" not in updated_parcels.columns:
            updated_parcels["gt_offset_m"] = np.nan

        offsets = []
        parcels_tree = STRtree(list(updated_parcels.geometry))

        for idx, pt_row in gt_points_gdf.iterrows():
            pt = pt_row.geometry
            nearest_idx = parcels_tree.nearest(pt)
            parcel_geom = updated_parcels.geometry.iloc[nearest_idx]

            # Distance to parcel boundary
            boundary_dist = float(parcel_geom.boundary.distance(pt))
            offsets.append(boundary_dist)

            curr_offset = updated_parcels.at[nearest_idx, "gt_offset_m"]
            if np.isnan(curr_offset):
                updated_parcels.at[nearest_idx, "gt_offset_m"] = boundary_dist
            else:
                updated_parcels.at[nearest_idx, "gt_offset_m"] = (curr_offset + boundary_dist) / 2.0

            # If boundary is close to ground survey point, mark verified
            if boundary_dist <= self.max_offset_tol:
                updated_parcels.at[nearest_idx, "status"] = "verified"
            else:
                updated_parcels.at[nearest_idx, "status"] = "disputed"

        mean_offset = float(np.mean(offsets)) if offsets else 0.0
        verified_count = int(np.sum(updated_parcels["status"] == "verified"))

        metrics = {
            "total_gt_points": len(gt_points_gdf),
            "mean_boundary_offset_m": round(mean_offset, 3),
            "verified_parcels_count": verified_count,
        }
        logger.info(f"GT matching completed: mean boundary offset {mean_offset:.3f}m, verified {verified_count} parcels.")
        return updated_parcels, metrics
