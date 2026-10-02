from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from ml.datasets.cadastra_dataset import CadastraDataset
from ml.datasets.dataset_builder import DatasetBuilder
from ml.datasets.label_rasterizer import LabelRasterizer
from tests.fixtures.synthetic_drone_data import generate_synthetic_drone_dataset
from backend.app.services.preprocessing.stacker import FeatureStackBuilder


@pytest.fixture
def synthetic_scene(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    data = generate_synthetic_drone_dataset(raw_dir, width=300, height=300, gsd=0.05)
    stack_path = tmp_path / "stack_4ch.tif"
    builder = FeatureStackBuilder(channel_preset="rgb+ndsm")
    builder.build_stack(data["ori"], data["dsm"], data["dtm"], stack_path)
    return {
        "stack": stack_path,
        "parcels": data["vector"],
    }


def test_label_rasterizer():
    rasterizer = LabelRasterizer(boundary_dilation_px=2, max_dist_px=20.0)
    transform = from_origin(0, 100, 1, 1)
    shape = (100, 100)

    # Create test polygon
    poly = Polygon([(20, 20), (60, 20), (60, 60), (20, 60), (20, 20)])
    gdf = gpd.GeoDataFrame({"id": ["p1"], "class": ["residential"]}, geometry=[poly])

    # 1. Test polygon interior rasterization
    mask = rasterizer.rasterize_polygons(gdf, shape, transform)
    assert mask.shape == shape
    assert mask[40, 40] == 1  # Inside polygon
    assert mask[5, 5] == 0    # Outside polygon

    # 2. Test boundary rasterization & distance transform
    boundary_mask, dist_map = rasterizer.rasterize_boundaries_and_distance(gdf, shape, transform)
    assert boundary_mask.shape == shape
    assert dist_map.shape == shape
    # Boundary dilated pixels should be 1
    assert np.any(boundary_mask > 0)
    # Distance map should peak at boundary
    assert np.max(dist_map) == 1.0
    assert np.min(dist_map) >= 0.0

    # 3. Test landuse rasterization
    lu_mask = rasterizer.rasterize_landuse(gdf, shape, transform)
    assert lu_mask.shape == shape
    assert lu_mask[40, 40] == LabelRasterizer.LANDUSE_CLASSES["residential"]


def test_dataset_builder_and_spatial_split(synthetic_scene, tmp_path: Path):
    ds_dir = tmp_path / "dataset"
    parcels_gdf = gpd.read_file(synthetic_scene["parcels"])

    builder = DatasetBuilder(chip_size_px=128, stride_px=96)
    manifest_path = builder.build_dataset(
        stack_path=synthetic_scene["stack"],
        output_dir=ds_dir,
        parcels_gdf=parcels_gdf,
    )

    assert manifest_path.exists()
    assert (ds_dir / "images").exists()
    assert (ds_dir / "masks").exists()

    # Load dataset via PyTorch Dataset
    train_ds = CadastraDataset(ds_dir, split="train", augment=True)
    assert len(train_ds) > 0

    img, targets = train_ds[0]
    assert img.shape == (4, 128, 128)
    assert targets["buildings"].shape == (1, 128, 128)
    assert targets["roads"].shape == (1, 128, 128)
    assert targets["boundaries"].shape == (1, 128, 128)
    assert targets["distance_map"].shape == (1, 128, 128)
    assert targets["landuse"].shape == (128, 128)
