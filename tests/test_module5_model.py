from pathlib import Path
import numpy as np
import pytest
import rasterio
import torch
from torch.optim import AdamW

from backend.app.services.preprocessing.stacker import FeatureStackBuilder
from ml.infer.tiled_infer import TiledPredictor
from ml.train.losses import MultiTaskCadastraLoss
from ml.train.metrics import MetricCalculator
from ml.train.model import MultiTaskCadastraNet
from tests.fixtures.synthetic_drone_data import generate_synthetic_drone_dataset


def test_multitask_model_forward():
    model = MultiTaskCadastraNet(in_channels=4, num_landuse_classes=9, pretrained=False)
    model.eval()

    # Batch of 2 samples, 4 channels, 128x128
    x = torch.randn(2, 4, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "buildings" in out
    assert "roads" in out
    assert "boundaries" in out
    assert "distance_map" in out
    assert "landuse" in out

    assert out["buildings"].shape == (2, 1, 128, 128)
    assert out["roads"].shape == (2, 1, 128, 128)
    assert out["boundaries"].shape == (2, 1, 128, 128)
    assert out["distance_map"].shape == (2, 1, 128, 128)
    assert out["landuse"].shape == (2, 9, 128, 128)


def test_multitask_loss():
    criterion = MultiTaskCadastraLoss()
    preds = {
        "buildings": torch.randn(2, 1, 64, 64),
        "roads": torch.randn(2, 1, 64, 64),
        "boundaries": torch.randn(2, 1, 64, 64),
        "distance_map": torch.sigmoid(torch.randn(2, 1, 64, 64)),
        "landuse": torch.randn(2, 9, 64, 64),
    }
    targets = {
        "buildings": torch.randint(0, 2, (2, 1, 64, 64)).float(),
        "roads": torch.randint(0, 2, (2, 1, 64, 64)).float(),
        "boundaries": torch.randint(0, 2, (2, 1, 64, 64)).float(),
        "distance_map": torch.rand(2, 1, 64, 64).float(),
        "landuse": torch.randint(0, 9, (2, 64, 64)).long(),
    }

    loss, breakdown = criterion(preds, targets)
    assert loss > 0.0
    assert "loss_buildings" in breakdown
    assert "loss_roads" in breakdown
    assert "loss_boundaries" in breakdown
    assert "loss_distance" in breakdown
    assert "loss_landuse" in breakdown


def test_boundary_f1_with_tolerance():
    # True boundary: horizontal line at row 50
    gt = np.zeros((100, 100), dtype=bool)
    gt[50, :] = True

    # Exact prediction
    exact_pred = gt.copy()
    m_exact = MetricCalculator.compute_boundary_f1_with_tolerance(exact_pred, gt, tolerance_px=3)
    assert m_exact["boundary_f1"] == pytest.approx(1.0)

    # Shifted prediction: row 51 (1 px shift, within 3px tolerance)
    shifted_pred = np.zeros((100, 100), dtype=bool)
    shifted_pred[51, :] = True
    m_shifted = MetricCalculator.compute_boundary_f1_with_tolerance(shifted_pred, gt, tolerance_px=3)
    assert m_shifted["boundary_f1"] == pytest.approx(1.0)

    # Far prediction: row 60 (10 px shift, outside 3px tolerance)
    far_pred = np.zeros((100, 100), dtype=bool)
    far_pred[60, :] = True
    m_far = MetricCalculator.compute_boundary_f1_with_tolerance(far_pred, gt, tolerance_px=3)
    assert m_far["boundary_f1"] == pytest.approx(0.0, abs=1e-3)


def test_overfit_on_tiny_batch():
    """Overfit smoke test: ensures loss strictly decreases on a fixed single batch."""
    torch.manual_seed(42)
    model = MultiTaskCadastraNet(in_channels=4, pretrained=False)
    criterion = MultiTaskCadastraLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(1, 4, 64, 64)
    targets = {
        "buildings": torch.zeros(1, 1, 64, 64),
        "roads": torch.zeros(1, 1, 64, 64),
        "boundaries": torch.zeros(1, 1, 64, 64),
        "distance_map": torch.zeros(1, 1, 64, 64),
        "landuse": torch.zeros(1, 64, 64).long(),
    }
    targets["buildings"][:, :, 10:30, 10:30] = 1.0
    targets["boundaries"][:, :, 20:25, :] = 1.0

    model.train()
    losses = []
    for step in range(8):
        optimizer.zero_grad()
        out = model(x)
        loss, _ = criterion(out, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: start {losses[0]} -> end {losses[-1]}"


def test_tiled_inference_synthetic(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    data = generate_synthetic_drone_dataset(raw_dir, width=128, height=128, gsd=0.05)
    stack_path = tmp_path / "stack_4ch.tif"
    builder = FeatureStackBuilder(channel_preset="rgb+ndsm")
    builder.build_stack(data["ori"], data["dsm"], data["dtm"], stack_path)

    model = MultiTaskCadastraNet(in_channels=4, pretrained=False)
    predictor = TiledPredictor(model=model, tile_size_px=64, overlap_ratio=0.25, use_tta=False)

    out_dir = tmp_path / "predictions"
    pred_paths = predictor.predict_raster(stack_path, out_dir)

    assert pred_paths["buildings"].exists()
    assert pred_paths["roads"].exists()
    assert pred_paths["boundaries"].exists()
    assert pred_paths["landuse"].exists()

    with rasterio.open(pred_paths["boundaries"]) as src:
        assert src.width == 128
        assert src.height == 128
        arr = src.read(1)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0
