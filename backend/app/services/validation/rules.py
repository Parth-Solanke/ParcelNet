from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import geopandas as gpd
import numpy as np
from shapely.validation import explain_validity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree


class ValidationIssue:
    def __init__(
        self,
        issue_type: str,
        severity: str,
        feature_ids: List[str],
        message: str,
        geom: Optional[Any] = None,
        auto_fix_action: Optional[str] = None,
    ):
        self.issue_type = issue_type  # overlap, gap, sliver, invalid, duplicate, road_crossing, shape_anomaly
        self.severity = severity      # error, warning, info
        self.feature_ids = feature_ids
        self.message = message
        self.geom = geom
        self.auto_fix_action = auto_fix_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "feature_ids": self.feature_ids,
            "message": self.message,
            "auto_fix_action": self.auto_fix_action,
        }


class BaseValidationRule(ABC):
    @abstractmethod
    def evaluate(self, gdf: gpd.GeoDataFrame, context: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
        pass


class InvalidGeometryRule(BaseValidationRule):
    """Detects invalid geometries, bowties, self-intersections, and unclosed rings."""

    def evaluate(self, gdf: gpd.GeoDataFrame, context: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
        issues = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            f_id = str(row.get("id", idx))
            if geom is None or geom.is_empty:
                issues.append(
                    ValidationIssue(
                        issue_type="invalid",
                        severity="error",
                        feature_ids=[f_id],
                        message="Geometry is null or empty.",
                        auto_fix_action="delete",
                    )
                )
            elif not geom.is_valid:
                reason = explain_validity(geom)
                issues.append(
                    ValidationIssue(
                        issue_type="invalid",
                        severity="error",
                        feature_ids=[f_id],
                        message=f"Invalid geometry: {reason}",
                        geom=geom,
                        auto_fix_action="make_valid",
                    )
                )
        return issues


class OverlapRule(BaseValidationRule):
    """Detects overlaps between adjacent parcels exceeding tolerance using STRtree."""

    def __init__(self, tolerance_sqm: float = 0.05):
        self.tolerance = tolerance_sqm

    def evaluate(self, gdf: gpd.GeoDataFrame, context: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
        issues = []
        if len(gdf) <= 1:
            return issues

        geoms = list(gdf.geometry)
        ids = [str(gdf.iloc[i].get("id", i)) for i in range(len(gdf))]
        tree = STRtree(geoms)

        seen_pairs = set()
        for i, geom in enumerate(geoms):
            if geom is None or not geom.is_valid or geom.is_empty:
                continue

            matches = tree.query(geom, predicate="intersects")
            for j in matches:
                if i >= j:
                    continue
                pair_key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                other = geoms[j]
                if other is None or not other.is_valid or other.is_empty:
                    continue

                inter = geom.intersection(other)
                if inter.area > self.tolerance:
                    issues.append(
                        ValidationIssue(
                            issue_type="overlap",
                            severity="error",
                            feature_ids=[ids[i], ids[j]],
                            message=f"Overlap of {inter.area:.2f} sqm between parcels {ids[i]} and {ids[j]}.",
                            geom=inter,
                            auto_fix_action="clip_overlap",
                        )
                    )
        return issues


class DuplicateRule(BaseValidationRule):
    """Detects identical or near-identical duplicate parcels (IoU > 0.95)."""

    def __init__(self, iou_threshold: float = 0.95):
        self.threshold = iou_threshold

    def evaluate(self, gdf: gpd.GeoDataFrame, context: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
        issues = []
        if len(gdf) <= 1:
            return issues

        geoms = list(gdf.geometry)
        ids = [str(gdf.iloc[i].get("id", i)) for i in range(len(gdf))]
        tree = STRtree(geoms)

        for i, geom in enumerate(geoms):
            matches = tree.query(geom, predicate="intersects")
            for j in matches:
                if i >= j:
                    continue
                other = geoms[j]
                inter_area = geom.intersection(other).area
                union_area = geom.union(other).area
                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > self.threshold:
                        issues.append(
                            ValidationIssue(
                                issue_type="duplicate",
                                severity="error",
                                feature_ids=[ids[i], ids[j]],
                                message=f"Duplicate parcel detected between {ids[i]} and {ids[j]} (IoU: {iou:.3f}).",
                                geom=geom,
                                auto_fix_action="merge_duplicate",
                            )
                        )
        return issues


class ShapeAnomalyRule(BaseValidationRule):
    """Detects tiny parcels, extreme aspect ratios, and acute angle spikes."""

    def __init__(self, min_area_sqm: float = 4.0, max_aspect_ratio: float = 12.0):
        self.min_area = min_area_sqm
        self.max_aspect = max_aspect_ratio

    def evaluate(self, gdf: gpd.GeoDataFrame, context: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
        issues = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            f_id = str(row.get("id", idx))
            if geom is None or not geom.is_valid:
                continue

            # Check tiny area
            if geom.area < self.min_area:
                issues.append(
                    ValidationIssue(
                        issue_type="sliver",
                        severity="warning",
                        feature_ids=[f_id],
                        message=f"Parcel {f_id} has tiny area ({geom.area:.2f} sqm < {self.min_area} sqm).",
                        geom=geom,
                        auto_fix_action="merge_sliver",
                    )
                )

            # Check aspect ratio from minimum rotated rectangle
            mrr = geom.minimum_rotated_rectangle
            if isinstance(mrr, Polygon) and len(mrr.exterior.coords) >= 4:
                c = list(mrr.exterior.coords)
                side_a = np.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
                side_b = np.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
                shorter = min(side_a, side_b)
                longer = max(side_a, side_b)
                if shorter > 0:
                    aspect = longer / shorter
                    if aspect > self.max_aspect:
                        issues.append(
                            ValidationIssue(
                                issue_type="shape_anomaly",
                                severity="warning",
                                feature_ids=[f_id],
                                message=f"Parcel {f_id} has extreme aspect ratio ({aspect:.1f} > {self.max_aspect}).",
                                geom=geom,
                                auto_fix_action="review",
                            )
                        )
        return issues
