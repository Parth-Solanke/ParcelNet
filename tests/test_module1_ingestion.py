from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from backend.app.services.ingestion.pipeline import IngestionPipeline
from backend.app.services.ingestion.validator import RasterValidator
from backend.app.services.ingestion.vector_ingest import VectorIngestor
from tests.fixtures.synthetic_drone_data import generate_synthetic_drone_dataset


@pytest.fixture
def synthetic_data(tmp_path: Path):
    data_dir = tmp_path / "raw_drone"
    return generate_synthetic_drone_dataset(data_dir, width=200, height=200, gsd=0.05)


def test_raster_validator_valid_files(synthetic_data):
    validator = RasterValidator(target_crs_epsg=32643)

    # Validate ORI
    ori_res = validator.validate_file(synthetic_data["ori"], "ORI")
    assert ori_res.is_valid is True
    assert ori_res.crs_epsg == 32643
    assert ori_res.count == 3
    assert ori_res.width == 200
    assert ori_res.height == 200
    assert len(ori_res.errors) == 0

    # Validate DSM
    dsm_res = validator.validate_file(synthetic_data["dsm"], "DSM")
    assert dsm_res.is_valid is True
    assert dsm_res.count == 1


def test_raster_validator_missing_crs(tmp_path: Path):
    # Create GeoTIFF without CRS
    bad_tif = tmp_path / "no_crs.tif"
    arr = np.zeros((50, 50), dtype=np.uint8)
    with rasterio.open(
        bad_tif,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=1,
        dtype="uint8",
        crs=None,  # No CRS
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(arr, 1)

    validator = RasterValidator(target_crs_epsg=32643)
    res = validator.validate_file(bad_tif, "ORI")
    assert res.is_valid is False
    assert any("has no defined CRS" in err for err in res.errors)


def test_bounding_overlap_calculation():
    bounds_a = (0.0, 0.0, 100.0, 100.0)
    bounds_b = (50.0, 0.0, 150.0, 100.0)
    overlap = RasterValidator.compute_overlap_ratio(bounds_a, bounds_b)
    # 50x100 overlap / 100x100 area = 0.5
    assert pytest.approx(overlap, 0.01) == 0.5

    # Completely disjoint
    bounds_c = (200.0, 200.0, 300.0, 300.0)
    assert RasterValidator.compute_overlap_ratio(bounds_a, bounds_c) == 0.0


def test_vector_ingest_repair_invalid_geometry(tmp_path: Path):
    # Create a bowtie (self-intersecting) polygon
    bowtie = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
    assert not bowtie.is_valid

    gdf = gpd.GeoDataFrame(
        {"id": ["bad_geom"]},
        geometry=[bowtie],
        crs="EPSG:32643",
    )
    raw_vec = tmp_path / "invalid_parcels.geojson"
    gdf.to_file(raw_vec, driver="GeoJSON")

    ingestor = VectorIngestor(target_crs_epsg=32643)
    repaired_gdf, summary = ingestor.ingest_vector_layer(raw_vec)

    assert summary.repaired_geometries_count == 1
    assert len(repaired_gdf) > 0
    assert all(geom.is_valid for geom in repaired_gdf.geometry)


def test_end_to_end_ingestion_pipeline(synthetic_data, tmp_path: Path):
    out_dir = tmp_path / "pipeline_out"
    pipeline = IngestionPipeline(target_crs_epsg=32643, output_dir=out_dir)

    result = pipeline.process(
        ori_path=synthetic_data["ori"],
        dsm_path=synthetic_data["dsm"],
        dtm_path=synthetic_data["dtm"],
        vector_path=synthetic_data["vector"],
    )

    assert "ORI" in result["processed_rasters"]
    assert "DSM" in result["processed_rasters"]
    assert "DTM" in result["processed_rasters"]
    assert result["qc_report"]["is_valid"] is True
    assert result["qc_report"]["alignment_verified"] is True
    assert Path(result["report_path"]).exists()
