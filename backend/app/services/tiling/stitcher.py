from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.windows import Window
from backend.app.core.logging import logger
from backend.app.services.tiling.blender import WindowBlender
from backend.app.services.tiling.window_tiler import TileMetadata


class TileStitcher:
    """
    Stitches overlapping probability or feature tiles back into a full-extent GeoTIFF
    using 2D tapering weight maps (Hann / Cosine) for seamless blending.
    """

    def __init__(
        self,
        full_width: int,
        full_height: int,
        channels: int,
        transform: rasterio.Affine,
        crs: rasterio.crs.CRS,
        blending_method: str = "hann",
        dtype: str = "float32",
    ):
        self.full_width = full_width
        self.full_height = full_height
        self.channels = channels
        self.transform = transform
        self.crs = crs
        self.blending_method = blending_method
        self.dtype = dtype

        # In-memory accumulators (channels x H x W and 1 x H x W for weights)
        self.accum_values = np.zeros((channels, full_height, full_width), dtype=np.float32)
        self.accum_weights = np.zeros((full_height, full_width), dtype=np.float32)

    def add_tile(
        self,
        tile_data: np.ndarray,
        window: Window,
        weight_map: Optional[np.ndarray] = None,
    ) -> None:
        """
        Adds a single tile chunk into the accumulated mosaic.
        tile_data shape: (channels, H, W) or (H, W).
        """
        if tile_data.ndim == 2:
            tile_data = tile_data[np.newaxis, ...]

        c, h, w = tile_data.shape
        col_off, row_off = int(window.col_off), int(window.row_off)

        if weight_map is None:
            if self.blending_method == "hann":
                weight_map = WindowBlender.get_hann_window_2d(h, w)
            else:
                weight_map = WindowBlender.get_cosine_window_2d(h, w)

        # Slice target bounds
        target_r_end = min(row_off + h, self.full_height)
        target_c_end = min(col_off + w, self.full_width)
        crop_h = target_r_end - row_off
        crop_w = target_c_end - col_off

        sub_tile = tile_data[:, :crop_h, :crop_w]
        sub_weight = weight_map[:crop_h, :crop_w]

        # Accumulate weighted values and weights
        self.accum_values[:, row_off:target_r_end, col_off:target_c_end] += (
            sub_tile * sub_weight[np.newaxis, ...]
        )
        self.accum_weights[row_off:target_r_end, col_off:target_c_end] += sub_weight

    def finalize(self) -> np.ndarray:
        """Normalizes accumulated values by accumulated weights to eliminate seam artifacts."""
        # Avoid division by zero in unreached corners
        safe_weights = np.maximum(self.accum_weights, 1e-6)
        normalized = self.accum_values / safe_weights[np.newaxis, ...]
        return normalized

    def save_to_geotiff(
        self,
        output_path: Path,
        compress: str = "DEFLATE",
        tiled: bool = True,
    ) -> Path:
        """Finalizes blended mosaic and writes to GeoTIFF."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mosaic = self.finalize()

        profile = {
            "driver": "GTiff",
            "height": self.full_height,
            "width": self.full_width,
            "count": self.channels,
            "dtype": self.dtype,
            "crs": self.crs,
            "transform": self.transform,
            "compress": compress,
            "tiled": tiled,
            "blockxsize": 512,
            "blockysize": 512,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            for idx in range(1, self.channels + 1):
                dst.write(mosaic[idx - 1].astype(self.dtype), idx)

        logger.info(f"Stitched blended mosaic saved to: {output_path}")
        return output_path
