from pathlib import Path
from typing import Dict, Tuple
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import Polygon


def generate_synthetic_drone_dataset(
    output_dir: Path,
    width: int = 500,
    height: int = 500,
    gsd: float = 0.05,
    epsg: int = 32643,
) -> Dict[str, Path]:
    """
    Generates synthetic drone GeoTIFFs (ORI, DSM, DTM) and vector parcels
    for deterministic unit testing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Origin in UTM coordinates (e.g. Easting 500000, Northing 2000000)
    origin_x = 500000.0
    origin_y = 2000000.0
    transform = from_origin(origin_x, origin_y, gsd, gsd)
    crs = CRS.from_epsg(epsg)

    # 1. Base terrain (gentle elevation gradient)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    base_terrain = 100.0 + (x_coords * 0.01) + (y_coords * 0.01)  # 100m to ~110m

    # 2. DTM = base terrain with slight microtopography
    dtm_data = base_terrain + (np.sin(x_coords / 20.0) * 0.2)

    # 3. Buildings and roads on top for DSM & ORI
    dsm_data = dtm_data.copy()
    r_band = np.full((height, width), 120, dtype=np.uint8)  # gray dirt/ground
    g_band = np.full((height, width), 140, dtype=np.uint8)
    b_band = np.full((height, width), 100, dtype=np.uint8)

    # Add a road corridor across the middle (horizontal road)
    road_y_start, road_y_end = 220, 260
    r_band[road_y_start:road_y_end, :] = 50
    g_band[road_y_start:road_y_end, :] = 50
    b_band[road_y_start:road_y_end, :] = 50

    # Add building 1 (north of road)
    b1_r1, b1_r2, b1_c1, b1_c2 = 50, 150, 60, 200
    dsm_data[b1_r1:b1_r2, b1_c1:b1_c2] += 8.5  # 8.5 meters high
    r_band[b1_r1:b1_r2, b1_c1:b1_c2] = 200  # Red roof
    g_band[b1_r1:b1_r2, b1_c1:b1_c2] = 60
    b_band[b1_r1:b1_r2, b1_c1:b1_c2] = 50

    # Add building 2 (south of road)
    b2_r1, b2_r2, b2_c1, b2_c2 = 300, 420, 280, 440
    dsm_data[b2_r1:b2_r2, b2_c1:b2_c2] += 12.0  # 12 meters high
    r_band[b2_r1:b2_r2, b2_c1:b2_c2] = 70  # Blue/slate roof
    g_band[b2_r1:b2_r2, b2_c1:b2_c2] = 100
    b_band[b2_r1:b2_r2, b2_c1:b2_c2] = 210

    # Add vegetation patch (green)
    veg_r1, veg_r2, veg_c1, veg_c2 = 50, 180, 320, 450
    dsm_data[veg_r1:veg_r2, veg_c1:veg_c2] += 2.0  # low vegetation/trees
    r_band[veg_r1:veg_r2, veg_c1:veg_c2] = 30
    g_band[veg_r1:veg_r2, veg_c1:veg_c2] = 180
    b_band[veg_r1:veg_r2, veg_c1:veg_c2] = 40

    # Write ORI (RGB)
    ori_path = output_dir / "synthetic_ori.tif"
    with rasterio.open(
        ori_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(r_band, 1)
        dst.write(g_band, 2)
        dst.write(b_band, 3)

    # Write DSM
    dsm_path = output_dir / "synthetic_dsm.tif"
    with rasterio.open(
        dsm_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(dsm_data.astype(np.float32), 1)

    # Write DTM
    dtm_path = output_dir / "synthetic_dtm.tif"
    with rasterio.open(
        dtm_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(dtm_data.astype(np.float32), 1)

    # Create vector parcels (GeoJSON)
    # Parcel 1 enclosing building 1
    p1_coords = [
        transform * (40, 30),
        transform * (220, 30),
        transform * (220, 200),
        transform * (40, 200),
        transform * (40, 30),
    ]
    # Parcel 2 enclosing building 2
    p2_coords = [
        transform * (260, 280),
        transform * (460, 280),
        transform * (460, 450),
        transform * (260, 450),
        transform * (260, 280),
    ]

    parcels_gdf = gpd.GeoDataFrame(
        {
            "id": ["p_001", "p_002"],
            "parcel_no": ["101/A", "102/B"],
            "geometry": [Polygon(p1_coords), Polygon(p2_coords)],
        },
        crs=crs,
    )
    vector_path = output_dir / "synthetic_parcels.geojson"
    parcels_gdf.to_file(vector_path, driver="GeoJSON")

    return {
        "ori": ori_path,
        "dsm": dsm_path,
        "dtm": dtm_path,
        "vector": vector_path,
    }
