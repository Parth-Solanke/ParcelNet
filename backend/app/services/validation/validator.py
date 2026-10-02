from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import geopandas as gpd
from shapely.geometry import Point
from backend.app.core.logging import logger
from backend.app.services.validation.rules import (
    BaseValidationRule,
    DuplicateRule,
    InvalidGeometryRule,
    OverlapRule,
    ShapeAnomalyRule,
    ValidationIssue,
)


class CadastraValidator:
    """Master validation service executing all topology and geometry rules."""

    def __init__(self, rules: Optional[List[BaseValidationRule]] = None):
        self.rules = rules or [
            InvalidGeometryRule(),
            OverlapRule(tolerance_sqm=0.05),
            DuplicateRule(iou_threshold=0.95),
            ShapeAnomalyRule(min_area_sqm=4.0, max_aspect_ratio=12.0),
        ]

    def validate(self, parcels_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Runs all validation rules and computes summary statistics."""
        all_issues: List[ValidationIssue] = []

        for rule in self.rules:
            issues = rule.evaluate(parcels_gdf)
            all_issues.extend(issues)

        # Categorize
        counts_by_type: Dict[str, int] = {}
        counts_by_severity: Dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        affected_features = set()

        for iss in all_issues:
            counts_by_type[iss.issue_type] = counts_by_type.get(iss.issue_type, 0) + 1
            counts_by_severity[iss.severity] = counts_by_severity.get(iss.severity, 0) + 1
            for fid in iss.feature_ids:
                affected_features.add(fid)

        total_parcels = len(parcels_gdf)
        clean_parcels = max(0, total_parcels - len(affected_features))
        clean_pct = round((clean_parcels / max(total_parcels, 1)) * 100.0, 2)

        summary = {
            "total_parcels": total_parcels,
            "clean_parcels": clean_parcels,
            "clean_percentage": clean_pct,
            "total_issues": len(all_issues),
            "counts_by_type": counts_by_type,
            "counts_by_severity": counts_by_severity,
            "issues": [iss.to_dict() for iss in all_issues],
        }

        # Build GeoDataFrame of issues
        issue_records = []
        for idx, iss in enumerate(all_issues):
            issue_geom = iss.geom
            if issue_geom is None or issue_geom.is_empty:
                issue_geom = Point(0, 0)

            issue_records.append({
                "id": f"issue_{idx+1:04d}",
                "type": iss.issue_type,
                "severity": iss.severity,
                "message": iss.message,
                "feature_ids": ",".join(iss.feature_ids),
                "auto_fix": iss.auto_fix_action or "none",
                "geometry": issue_geom,
            })

        if issue_records:
            issues_gdf = gpd.GeoDataFrame(issue_records, crs=parcels_gdf.crs)
        else:
            issues_gdf = gpd.GeoDataFrame(
                columns=["id", "type", "severity", "message", "feature_ids", "auto_fix", "geometry"],
                geometry="geometry",
                crs=parcels_gdf.crs,
            )
        return {"summary": summary, "issues_gdf": issues_gdf}
