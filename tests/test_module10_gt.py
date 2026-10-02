from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from backend.app.services.gt_integration.importer import GTImporter
from backend.app.services.gt_integration.matcher import GTMatcher
from backend.app.services.gt_integration.verification_list import FieldVerificationExporter


def test_gt_importer_csv(tmp_path: Path):
    csv_file = tmp_path / "control_points.csv"
    df = pd.DataFrame({
        "point_id": ["CP_01", "CP_02"],
        "easting": [500010.5, 500050.2],
        "northing": [2000010.8, 2000045.1],
        "height": [105.2, 106.1],
        "accuracy_cm": [2.5, 3.1],
    })
    df.to_csv(csv_file, index=False)

    importer = GTImporter(target_crs_epsg=32643)
    gdf = importer.import_points_csv(csv_file, x_col="easting", y_col="northing")

    assert len(gdf) == 2
    assert gdf.crs.to_string() == "EPSG:32643"
    assert isinstance(gdf.geometry.iloc[0], Point)


def test_gt_matcher_and_verification_list(tmp_path: Path):
    # Parcel from (0,0) to (10,10)
    parcel = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    parcels_gdf = gpd.GeoDataFrame(
        {"id": ["P101"], "confidence": [0.60]},
        geometry=[parcel],
        crs="EPSG:32643",
    )

    # GT point right on the corner (0.05m offset)
    gt_pt = Point(10.05, 10.0)
    gt_gdf = gpd.GeoDataFrame(
        {"pt_id": ["GT1"]},
        geometry=[gt_pt],
        crs="EPSG:32643",
    )

    matcher = GTMatcher(max_boundary_offset_tolerance_m=0.20)
    matched_parcels, metrics = matcher.match_points_to_parcels(parcels_gdf, gt_gdf)

    assert metrics["verified_parcels_count"] == 1
    assert matched_parcels.iloc[0]["status"] == "verified"
    assert matched_parcels.iloc[0]["gt_offset_m"] < 0.10

    # Test Field Verification Exporter
    exporter = FieldVerificationExporter(confidence_threshold=0.65)
    # Give a parcel low confidence
    parcels_gdf.at[0, "status"] = "disputed"
    v_list = exporter.create_verification_list(parcels_gdf, output_dir=tmp_path / "worklist")

    assert len(v_list) == 1
    assert "surveyor_name" in v_list.columns
    assert (tmp_path / "worklist" / "field_verification_list.geojson").exists()
