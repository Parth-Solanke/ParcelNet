from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from backend.app.services.landuse.classifier import ObjectBasedLandUseRefiner


@pytest.fixture
def synthetic_landuse_setup(tmp_path: Path):
    transform = from_origin(500000, 2000000, 1.0, 1.0)
    crs = "EPSG:32643"
    h, w = 50, 50

    # Land-use raster: residential (1) on top, vegetation (5) on bottom
    lu_arr = np.full((h, w), 5, dtype=np.uint8)  # default vegetation
    lu_arr[:25, :] = 1  # residential
    lu_path = tmp_path / "landuse.tif"
    with rasterio.open(
        lu_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8", crs=crs, transform=transform
    ) as dst:
        dst.write(lu_arr, 1)

    # nDSM raster: tall structure on top (10m)
    ndsm_arr = np.zeros((h, w), dtype=np.float32)
    ndsm_arr[:25, :] = 10.0
    ndsm_path = tmp_path / "ndsm.tif"
    with rasterio.open(
        ndsm_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs, transform=transform
    ) as dst:
        dst.write(ndsm_arr, 1)

    # 2 Parcels
    p1 = Polygon([
        transform * (5, 5),
        transform * (20, 5),
        transform * (20, 20),
        transform * (5, 20),
        transform * (5, 5),
    ])
    p2 = Polygon([
        transform * (5, 30),
        transform * (20, 30),
        transform * (20, 45),
        transform * (5, 45),
        transform * (5, 30),
    ])

    parcels_gdf = gpd.GeoDataFrame(
        {"id": ["p1", "p2"]},
        geometry=[p1, p2],
        crs=crs,
    )

    return {"parcels": parcels_gdf, "landuse": lu_path, "ndsm": ndsm_path}


def test_object_based_landuse_refinement(synthetic_landuse_setup):
    refiner = ObjectBasedLandUseRefiner(tall_building_height_m=3.5)
    result_gdf = refiner.refine_parcels(
        parcels_gdf=synthetic_landuse_setup["parcels"],
        landuse_raster_path=synthetic_landuse_setup["landuse"],
        ndsm_raster_path=synthetic_landuse_setup["ndsm"],
    )

    assert "landuse" in result_gdf.columns
    assert "lu_confidence" in result_gdf.columns
    assert result_gdf.loc[result_gdf["id"] == "p1", "landuse"].iloc[0] == "residential"
    assert result_gdf.loc[result_gdf["id"] == "p2", "landuse"].iloc[0] == "vegetation"
