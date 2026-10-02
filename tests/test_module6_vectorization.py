from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from backend.app.services.vectorization.building_vectorizer import BuildingVectorizer
from backend.app.services.vectorization.road_vectorizer import RoadVectorizer


@pytest.fixture
def synthetic_prob_rasters(tmp_path: Path):
    transform = from_origin(500000, 2000000, 0.05, 0.05)
    crs = "EPSG:32643"
    h, w = 150, 150

    # Building probability raster: 1 rectangular building
    b_prob = np.zeros((h, w), dtype=np.float32)
    b_prob[30:70, 30:80] = 0.95
    b_path = tmp_path / "b_prob.tif"
    with rasterio.open(
        b_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs, transform=transform
    ) as dst:
        dst.write(b_prob, 1)

    # Road probability raster: 1 horizontal road strip
    r_prob = np.zeros((h, w), dtype=np.float32)
    r_prob[90:110, :] = 0.90
    r_path = tmp_path / "r_prob.tif"
    with rasterio.open(
        r_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs, transform=transform
    ) as dst:
        dst.write(r_prob, 1)

    # nDSM raster
    ndsm = np.zeros((h, w), dtype=np.float32)
    ndsm[30:70, 30:80] = 7.5
    ndsm_path = tmp_path / "ndsm.tif"
    with rasterio.open(
        ndsm_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs, transform=transform
    ) as dst:
        dst.write(ndsm, 1)

    return {"bldg": b_path, "road": r_path, "ndsm": ndsm_path}


def test_building_vectorizer(synthetic_prob_rasters):
    vec = BuildingVectorizer(prob_threshold=0.50, min_area_sqm=4.0)
    gdf = vec.vectorize(synthetic_prob_rasters["bldg"], ndsm_path=synthetic_prob_rasters["ndsm"])

    assert len(gdf) >= 1
    assert "area_sqm" in gdf.columns
    assert "height_m" in gdf.columns
    assert "confidence" in gdf.columns
    assert gdf.iloc[0]["height_m"] > 5.0  # Captured median height
    assert gdf.iloc[0]["geometry"].is_valid


def test_building_orthogonalization():
    vec = BuildingVectorizer()
    # Create slightly skewed rectangle
    coords = [(0, 0), (10, 0.5), (9.5, 10), (0, 9.8), (0, 0)]
    poly = Polygon(coords)
    ortho = vec.orthogonalize_polygon(poly)
    assert ortho.is_valid
    assert ortho.area > 0


def test_road_vectorizer(synthetic_prob_rasters):
    r_vec = RoadVectorizer(prob_threshold=0.50, min_spur_length_m=1.0)
    c_lines, polys, graph = r_vec.vectorize(synthetic_prob_rasters["road"])

    assert len(c_lines) >= 1
    assert len(polys) >= 1
    assert graph.number_of_nodes() >= 2
    assert "width_m" in c_lines.columns
    assert c_lines.iloc[0]["geometry"].is_valid
