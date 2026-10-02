import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon
import pytest

from scripts.evaluate import evaluate_parcels, generate_html_report


def test_evaluate_parcels(tmp_path):
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

    # Reference polygons
    ref_gdf = gpd.GeoDataFrame(
        [{"id": 1, "geometry": p1}, {"id": 2, "geometry": p2}],
        crs="EPSG:32643",
    )

    # Predicted polygons with minor perturbation
    p1_pred = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p2_pred = Polygon([(10.1, 0), (20.1, 0), (20.1, 10), (10.1, 10)])
    pred_gdf = gpd.GeoDataFrame(
        [{"id": 101, "geometry": p1_pred}, {"id": 102, "geometry": p2_pred}],
        crs="EPSG:32643",
    )

    metrics = evaluate_parcels(pred_gdf, ref_gdf)
    assert metrics["parcels_count"] == 2
    assert metrics["mean_iou"] > 0.80
    assert metrics["completeness"] == 1.0
    assert metrics["correctness"] == 1.0
    assert metrics["clean_parcels_pct"] == 100.0
    assert "manual_hours_saved" in metrics

    # HTML report generation
    html_path = tmp_path / "evaluation_report.html"
    generate_html_report(metrics, html_path)
    assert html_path.exists()
    assert "CadastraAI Evaluation & Benchmarking Report" in html_path.read_text(encoding="utf-8")


def test_evaluate_parcels_empty():
    empty_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:32643")
    metrics = evaluate_parcels(empty_gdf, empty_gdf)
    assert metrics["iou"] == 0.0
    assert metrics["completeness"] == 0.0
