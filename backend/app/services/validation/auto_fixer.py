from typing import Any, Dict, List, Tuple
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid
from backend.app.core.logging import logger


class AutoFixEngine:
    """Automated geometry and topology healing engine."""

    def __init__(self, tolerance_overlap_sqm: float = 0.05):
        self.tol_overlap = tolerance_overlap_sqm

    def auto_fix_all(self, parcels_gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, List[Dict[str, Any]]]:
        """
        Applies sequential healing:
        1. make_valid on invalid shapes
        2. Drops duplicates
        3. Clips overlaps to higher-confidence parcel
        """
        fixed_gdf = parcels_gdf.copy()
        audit_log: List[Dict[str, Any]] = []

        # 1. Fix Invalid
        for idx in range(len(fixed_gdf)):
            geom = fixed_gdf.geometry.iloc[idx]
            if geom is not None and not geom.is_valid:
                repaired = make_valid(geom)
                if isinstance(repaired, (Polygon, MultiPolygon)):
                    fixed_gdf.loc[idx, "geometry"] = repaired
                    audit_log.append({
                        "feature_id": str(fixed_gdf.iloc[idx].get("id", idx)),
                        "action": "make_valid",
                        "status": "repaired",
                    })

        # 2. Fix Duplicates
        keep_indices = set(range(len(fixed_gdf)))
        tree = STRtree(list(fixed_gdf.geometry))
        for i in range(len(fixed_gdf)):
            if i not in keep_indices:
                continue
            geom = fixed_gdf.geometry.iloc[i]
            matches = tree.query(geom, predicate="intersects")
            for j in matches:
                if i >= j or j not in keep_indices:
                    continue
                other = fixed_gdf.geometry.iloc[j]
                inter_area = geom.intersection(other).area
                union_area = geom.union(other).area
                if union_area > 0 and (inter_area / union_area) > 0.95:
                    # Drop parcel with lower confidence
                    conf_i = fixed_gdf.iloc[i].get("confidence", 0.5)
                    conf_j = fixed_gdf.iloc[j].get("confidence", 0.5)
                    drop_idx = j if conf_i >= conf_j else i
                    keep_indices.discard(drop_idx)
                    audit_log.append({
                        "feature_id": str(fixed_gdf.iloc[drop_idx].get("id", drop_idx)),
                        "action": "drop_duplicate",
                        "status": "removed",
                    })

        fixed_gdf = fixed_gdf.iloc[sorted(list(keep_indices))].copy().reset_index(drop=True)

        # 3. Clip Overlaps (favor higher confidence)
        tree = STRtree(list(fixed_gdf.geometry))
        for i in range(len(fixed_gdf)):
            geom_i = fixed_gdf.geometry.iloc[i]
            if geom_i is None or not geom_i.is_valid:
                continue

            matches = tree.query(geom_i, predicate="intersects")
            for j in matches:
                if i >= j or j >= len(fixed_gdf):
                    continue
                geom_j = fixed_gdf.geometry.iloc[j]
                if geom_j is None or not geom_j.is_valid:
                    continue

                inter = geom_i.intersection(geom_j)
                if inter.area > self.tol_overlap:
                    conf_i = fixed_gdf.iloc[i].get("confidence", 0.5)
                    conf_j = fixed_gdf.iloc[j].get("confidence", 0.5)

                    if conf_i >= conf_j:
                        clipped_j = geom_j.difference(geom_i)
                        if isinstance(clipped_j, (Polygon, MultiPolygon)) and not clipped_j.is_empty:
                            fixed_gdf.loc[j, "geometry"] = clipped_j
                            audit_log.append({
                                "feature_id": str(fixed_gdf.iloc[j].get("id", j)),
                                "action": "clip_overlap",
                                "clipped_against": str(fixed_gdf.iloc[i].get("id", i)),
                            })
                    else:
                        clipped_i = geom_i.difference(geom_j)
                        if isinstance(clipped_i, (Polygon, MultiPolygon)) and not clipped_i.is_empty:
                            fixed_gdf.loc[i, "geometry"] = clipped_i
                            geom_i = clipped_i
                            audit_log.append({
                                "feature_id": str(fixed_gdf.iloc[i].get("id", i)),
                                "action": "clip_overlap",
                                "clipped_against": str(fixed_gdf.iloc[j].get("id", j)),
                            })

        logger.info(f"AutoFixEngine applied {len(audit_log)} topology and geometry repairs.")
        return fixed_gdf, audit_log
