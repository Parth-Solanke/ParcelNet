import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from backend.app.services.validation.auto_fixer import AutoFixEngine
from backend.app.services.validation.rules import (
    DuplicateRule,
    InvalidGeometryRule,
    OverlapRule,
    ShapeAnomalyRule,
)
from backend.app.services.validation.validator import CadastraValidator


def test_invalid_geometry_rule():
    bowtie = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
    gdf = gpd.GeoDataFrame({"id": ["bowtie_poly"]}, geometry=[bowtie], crs="EPSG:32643")

    rule = InvalidGeometryRule()
    issues = rule.evaluate(gdf)
    assert len(issues) == 1
    assert issues[0].issue_type == "invalid"
    assert issues[0].severity == "error"


def test_overlap_rule():
    # Two parcels overlapping by 20 sqm
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    p2 = Polygon([(8, 0), (18, 0), (18, 10), (8, 10), (8, 0)])  # 2x10 = 20 sqm overlap
    gdf = gpd.GeoDataFrame({"id": ["p1", "p2"]}, geometry=[p1, p2], crs="EPSG:32643")

    rule = OverlapRule(tolerance_sqm=0.05)
    issues = rule.evaluate(gdf)
    assert len(issues) == 1
    assert issues[0].issue_type == "overlap"
    assert "p1" in issues[0].feature_ids and "p2" in issues[0].feature_ids


def test_duplicate_rule():
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    p2 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])  # Identical
    gdf = gpd.GeoDataFrame({"id": ["p1", "p2"]}, geometry=[p1, p2], crs="EPSG:32643")

    rule = DuplicateRule(iou_threshold=0.95)
    issues = rule.evaluate(gdf)
    assert len(issues) == 1
    assert issues[0].issue_type == "duplicate"


def test_shape_anomaly_rule():
    # Needle shape: 100m long, 1m wide -> aspect ratio 100
    needle = Polygon([(0, 0), (100, 0), (100, 1), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame({"id": ["needle"]}, geometry=[needle], crs="EPSG:32643")

    rule = ShapeAnomalyRule(min_area_sqm=4.0, max_aspect_ratio=10.0)
    issues = rule.evaluate(gdf)
    assert any(iss.issue_type == "shape_anomaly" for iss in issues)


def test_auto_fix_engine_overlap_and_invalid():
    bowtie = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
    p_high = Polygon([(10, 0), (20, 0), (20, 10), (10, 10), (10, 0)])
    p_low = Polygon([(18, 0), (28, 0), (28, 10), (18, 10), (18, 0)])  # Overlaps p_high

    gdf = gpd.GeoDataFrame(
        {
            "id": ["p_bowtie", "p_high", "p_low"],
            "confidence": [0.5, 0.95, 0.60],
        },
        geometry=[bowtie, p_high, p_low],
        crs="EPSG:32643",
    )

    fixer = AutoFixEngine(tolerance_overlap_sqm=0.05)
    healed_gdf, audit = fixer.auto_fix_all(gdf)

    # 1. Invalid geometry repaired
    assert all(geom.is_valid for geom in healed_gdf.geometry)

    # 2. Overlap eliminated
    geom_high = healed_gdf.loc[healed_gdf["id"] == "p_high", "geometry"].iloc[0]
    geom_low = healed_gdf.loc[healed_gdf["id"] == "p_low", "geometry"].iloc[0]
    inter_area = geom_high.intersection(geom_low).area
    assert inter_area <= 0.05


def test_master_cadastra_validator():
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    gdf = gpd.GeoDataFrame({"id": ["clean_parcel"]}, geometry=[p1], crs="EPSG:32643")

    validator = CadastraValidator()
    report = validator.validate(gdf)
    summary = report["summary"]

    assert summary["clean_percentage"] == 100.0
    assert summary["total_issues"] == 0
