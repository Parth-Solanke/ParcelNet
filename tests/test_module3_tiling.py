from pathlib import Path
import numpy as np
import pytest
import rasterio

from backend.app.services.tiling.blender import WindowBlender
from backend.app.services.tiling.stitcher import TileStitcher
from backend.app.services.tiling.window_tiler import WindowTiler
from tests.fixtures.synthetic_drone_data import generate_synthetic_drone_dataset


@pytest.fixture
def synthetic_data(tmp_path: Path):
    data_dir = tmp_path / "raw_drone"
    return generate_synthetic_drone_dataset(data_dir, width=240, height=240, gsd=0.05)


def test_window_blender_properties():
    h, w = 64, 64
    hann = WindowBlender.get_hann_window_2d(h, w)
    assert hann.shape == (h, w)
    assert hann.max() <= 1.0
    assert hann.min() >= 1e-4
    # Center pixel should be maximum
    assert hann[h // 2, w // 2] == pytest.approx(1.0, abs=1e-2)

    # Cosine window
    cos_w = WindowBlender.get_cosine_window_2d(h, w)
    assert cos_w.shape == (h, w)
    assert cos_w.max() <= 1.0


def test_sliding_window_tiler_coverage(synthetic_data):
    tiler = WindowTiler(tile_size_px=64, overlap_ratio=0.25)
    with rasterio.open(synthetic_data["ori"]) as src:
        windows = tiler.compute_tile_windows(src.width, src.height, src.transform)
        assert len(windows) > 1

        # Check coverage of every pixel
        coverage_mask = np.zeros((src.height, src.width), dtype=int)
        for win, _ in windows:
            coverage_mask[
                int(win.row_off) : int(win.row_off + win.height),
                int(win.col_off) : int(win.col_off + win.width),
            ] += 1

        # Every pixel must be covered at least once
        assert np.all(coverage_mask >= 1)


def test_tile_and_stitch_identity(synthetic_data, tmp_path: Path):
    """
    Core Acceptance Test:
    Tiling an original raster with overlap and re-stitching using Hann blending
    must reconstruct the original raster with near-zero error.
    """
    tile_size = 64
    overlap = 0.25
    tiler = WindowTiler(tile_size_px=tile_size, overlap_ratio=overlap)

    with rasterio.open(synthetic_data["ori"]) as src:
        orig_data = src.read().astype(np.float32)
        stitcher = TileStitcher(
            full_width=src.width,
            full_height=src.height,
            channels=src.count,
            transform=src.transform,
            crs=src.crs,
            blending_method="hann",
            dtype="float32",
        )

        for tile_data, win, meta in tiler.iter_tiles(synthetic_data["ori"]):
            stitcher.add_tile(tile_data.astype(np.float32), win)

        reconstructed = stitcher.finalize()

    # Verify shapes match
    assert reconstructed.shape == orig_data.shape

    # Max absolute difference across all pixels and bands should be minimal
    diff = np.abs(reconstructed - orig_data)
    max_error = np.max(diff)
    mean_error = np.mean(diff)

    # Tolerance: original is uint8 [0, 255], numerical error in float32 weighted reconstruction is < 0.05
    assert max_error < 0.1, f"Max reconstruction error too high: {max_error}"
    assert mean_error < 0.01, f"Mean reconstruction error too high: {mean_error}"


def test_tile_index_geojson_export(synthetic_data, tmp_path: Path):
    tiler = WindowTiler(tile_size_px=100, overlap_ratio=0.20)
    out_geojson = tmp_path / "tile_index.geojson"
    result_path = tiler.export_tile_index_geojson(synthetic_data["ori"], out_geojson)

    assert result_path.exists()
    with open(result_path, "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0
        assert "tile_id" in data["features"][0]["properties"]
