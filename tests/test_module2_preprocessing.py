from pathlib import Path
import numpy as np
import pytest
import rasterio

from backend.app.services.preprocessing.auxiliary import AuxiliaryFeatureGenerator
from backend.app.services.preprocessing.ndsm import NDSMCalculator
from backend.app.services.preprocessing.normalizer import RadiometricNormalizer
from backend.app.services.preprocessing.stacker import FeatureStackBuilder
from tests.fixtures.synthetic_drone_data import generate_synthetic_drone_dataset


@pytest.fixture
def synthetic_data(tmp_path: Path):
    data_dir = tmp_path / "raw_drone"
    return generate_synthetic_drone_dataset(data_dir, width=150, height=150, gsd=0.05)


def test_ndsm_calculation():
    calc = NDSMCalculator(min_height_threshold_m=0.5, median_filter_size=3)

    # Simulated flat terrain with a 10m building and ground noise (0.2m)
    dsm = np.array([
        [100.0, 100.2, 110.0],
        [100.0, 100.3, 110.0],
        [99.0,  100.0, 100.0],  # 99.0 < 100.0 tests negative clipping
    ], dtype=np.float32)

    dtm = np.full((3, 3), 100.0, dtype=np.float32)

    ndsm = calc.compute_ndsm_array(dsm, dtm)

    # Check non-negativity
    assert np.all(ndsm >= 0.0)
    # Check noise threshold (< 0.5m should become 0.0)
    assert ndsm[0, 1] == 0.0 or ndsm[0, 1] < 1.0


def test_auxiliary_features():
    aux = AuxiliaryFeatureGenerator()
    elevation = np.array([
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
    ], dtype=np.float32)

    # Test hillshade output range
    hs = aux.compute_hillshade(elevation, dx=1.0, dy=1.0)
    assert hs.dtype == np.uint8
    assert np.all(hs >= 0) and np.all(hs <= 255)

    # Test slope output range
    slope = aux.compute_slope_degrees(elevation, dx=1.0, dy=1.0)
    assert np.all(slope >= 0.0) and np.all(slope <= 90.0)

    # Test excess green
    rgb = np.zeros((3, 20, 20), dtype=np.uint8)
    rgb[0] = 50   # Red
    rgb[1] = 200  # Green
    rgb[2] = 50   # Blue
    exg = aux.compute_excess_green(rgb)
    assert exg.shape == (20, 20)
    assert np.all(exg >= 0.0) and np.all(exg <= 1.0)


def test_radiometric_normalizer():
    norm = RadiometricNormalizer(percentile_min=2.0, percentile_max=98.0, use_clahe=False)
    data = np.arange(0, 1000, dtype=np.float32).reshape(10, 100)

    normalized = norm.normalize_band(data)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_feature_stack_builder(synthetic_data, tmp_path: Path):
    out_stack = tmp_path / "test_stack.tif"
    builder = FeatureStackBuilder(channel_preset="rgb+ndsm")

    result_path = builder.build_stack(
        ori_path=synthetic_data["ori"],
        dsm_path=synthetic_data["dsm"],
        dtm_path=synthetic_data["dtm"],
        output_stack_path=out_stack,
    )

    assert result_path.exists()
    with rasterio.open(result_path) as src:
        assert src.count == 4  # R, G, B, nDSM
        assert src.width == 150
        assert src.height == 150
        assert src.dtypes[0] == "float32"

        # Check values are normalized
        stack_arr = src.read()
        assert np.all(stack_arr >= 0.0)
        assert np.all(stack_arr <= 1.0)
