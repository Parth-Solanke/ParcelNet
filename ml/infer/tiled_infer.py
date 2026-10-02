from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import rasterio
import torch

from backend.app.core.logging import logger
from backend.app.services.tiling.stitcher import TileStitcher
from backend.app.services.tiling.window_tiler import WindowTiler
from ml.train.model import MultiTaskCadastraNet


class TiledPredictor:
    """Performs sliding-window tiled inference with Test-Time Augmentation (TTA) and Hann seam blending."""

    def __init__(
        self,
        model: MultiTaskCadastraNet,
        tile_size_px: int = 1024,
        overlap_ratio: float = 0.25,
        use_tta: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tile_size = tile_size_px
        self.overlap_ratio = overlap_ratio
        self.use_tta = use_tta
        self.tiler = WindowTiler(tile_size_px=tile_size_px, overlap_ratio=overlap_ratio)

    def _predict_chip(self, chip_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Predicts on a single chip (1, C, H, W) with optional test-time augmentation (TTA)."""
        with torch.no_grad():
            preds = self.model(chip_tensor)

            p_bldg = torch.sigmoid(preds["buildings"])
            p_road = torch.sigmoid(preds["roads"])
            p_bound = torch.sigmoid(preds["boundaries"])
            p_lu = torch.softmax(preds["landuse"], dim=1)

            if self.use_tta:
                # Horizontal flip
                chip_h = torch.flip(chip_tensor, dims=[3])
                preds_h = self.model(chip_h)
                p_bldg = (p_bldg + torch.flip(torch.sigmoid(preds_h["buildings"]), dims=[3])) / 2.0
                p_road = (p_road + torch.flip(torch.sigmoid(preds_h["roads"]), dims=[3])) / 2.0
                p_bound = (p_bound + torch.flip(torch.sigmoid(preds_h["boundaries"]), dims=[3])) / 2.0
                p_lu = (p_lu + torch.flip(torch.softmax(preds_h["landuse"], dim=1), dims=[3])) / 2.0

                # Vertical flip
                chip_v = torch.flip(chip_tensor, dims=[2])
                preds_v = self.model(chip_v)
                p_bldg = (p_bldg * 2.0 + torch.flip(torch.sigmoid(preds_v["buildings"]), dims=[2])) / 3.0
                p_road = (p_road * 2.0 + torch.flip(torch.sigmoid(preds_v["roads"]), dims=[2])) / 3.0
                p_bound = (p_bound * 2.0 + torch.flip(torch.sigmoid(preds_v["boundaries"]), dims=[2])) / 3.0
                p_lu = (p_lu * 2.0 + torch.flip(torch.softmax(preds_v["landuse"], dim=1), dims=[2])) / 3.0

        return {
            "buildings": p_bldg[0, 0].cpu().numpy(),
            "roads": p_road[0, 0].cpu().numpy(),
            "boundaries": p_bound[0, 0].cpu().numpy(),
            "landuse": torch.argmax(p_lu, dim=1)[0].cpu().numpy().astype(np.float32),
        }

    def predict_raster(
        self,
        stack_path: Path,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Runs full tiled inference over 4-channel raster stack and outputs probability maps."""
        output_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(stack_path) as src:
            width, height = src.width, src.height
            transform = src.transform
            crs = src.crs

            stitchers = {
                "buildings": TileStitcher(width, height, 1, transform, crs, dtype="float32"),
                "roads": TileStitcher(width, height, 1, transform, crs, dtype="float32"),
                "boundaries": TileStitcher(width, height, 1, transform, crs, dtype="float32"),
                "landuse": TileStitcher(width, height, 1, transform, crs, dtype="float32"),
            }

            logger.info(f"Running tiled inference on {stack_path.name} ({width}x{height} px)...")
            for tile_data, win, meta in self.tiler.iter_tiles(stack_path):
                # Ensure 4-channel input
                if tile_data.shape[0] < 4:
                    padded = np.zeros((4, tile_data.shape[1], tile_data.shape[2]), dtype=np.float32)
                    padded[: tile_data.shape[0]] = tile_data
                    tile_data = padded

                chip_t = torch.from_numpy(tile_data.astype(np.float32)).unsqueeze(0).to(self.device)
                chip_preds = self._predict_chip(chip_t)

                for head_name, pred_arr in chip_preds.items():
                    stitchers[head_name].add_tile(pred_arr, win)

            # Save probability GeoTIFFs
            output_paths: Dict[str, Path] = {}
            for head_name, stitcher in stitchers.items():
                out_path = output_dir / f"{head_name}_prob.tif"
                stitcher.save_to_geotiff(out_path)
                output_paths[head_name] = out_path

        logger.info(f"Tiled inference complete! Output maps saved to: {output_dir}")
        return output_paths
