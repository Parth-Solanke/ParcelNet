from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from backend.app.services.topology.line_network import BoundaryLineNetworkBuilder
from backend.app.services.topology.polygonizer import FacePolygonizer


def test_face_polygonizer_basic():
    polygonizer = FacePolygonizer(min_area_sqm=5.0)

    # 4 lines forming a closed 10x10 square + a diagonal splitting it into 2 triangles
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(10, 0), (10, 10)]),
        LineString([(10, 10), (0, 10)]),
        LineString([(0, 10), (0, 0)]),
        LineString([(0, 0), (10, 10)]),
    ]

    gdf = polygonizer.polygonize_network(lines)
    assert len(gdf) == 2
    assert all(p.is_valid for p in gdf.geometry)
    assert "compactness" in gdf.columns
    assert "confidence" in gdf.columns
    assert "area_sqm" in gdf.columns


def test_sliver_merging():
    polygonizer = FacePolygonizer(min_area_sqm=2.0, sliver_area_sqm=10.0, sliver_compactness_threshold=0.2)

    # Large main parcel (100 sqm) + adjacent thin sliver (3 sqm)
    main_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    sliver_poly = Polygon([(10, 0), (10.3, 0), (10.3, 10), (10, 10), (10, 0)])

    merged = polygonizer.merge_slivers([main_poly, sliver_poly])
    assert len(merged) == 1
    assert merged[0].area == pytest.approx(103.0, abs=0.1)


def test_boundary_line_network_builder(tmp_path: Path):
    builder = BoundaryLineNetworkBuilder(prob_threshold=0.30, min_spur_length_m=1.0)
    transform = from_origin(500000, 2000000, 1.0, 1.0)
    crs = "EPSG:32643"
    h, w = 50, 50

    # Draw a square boundary outline in prob raster
    prob = np.zeros((h, w), dtype=np.float32)
    prob[10, 10:40] = 0.9  # top
    prob[40, 10:40] = 0.9  # bottom
    prob[10:41, 10] = 0.9  # left
    prob[10:41, 39] = 0.9  # right

    prob_path = tmp_path / "bound_prob.tif"
    with rasterio.open(
        prob_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs, transform=transform
    ) as dst:
        dst.write(prob, 1)

    noded_lines = builder.build_network(prob_path)
    assert len(noded_lines) >= 1
