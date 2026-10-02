from typing import Optional, Tuple
import cv2
import numpy as np


class RadiometricNormalizer:
    """Performs radiometric normalization (percentile stretch 2-98% and optional CLAHE)."""

    def __init__(
        self,
        percentile_min: float = 2.0,
        percentile_max: float = 98.0,
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
    ):
        self.p_min = percentile_min
        self.p_max = percentile_max
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

    def normalize_band(
        self,
        band: np.ndarray,
        nodata_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalizes a single band to float32 [0.0, 1.0] using percentile stretch."""
        valid_pixels = band[~nodata_mask] if nodata_mask is not None else band.flatten()
        if len(valid_pixels) == 0:
            return np.zeros_like(band, dtype=np.float32)

        v_min = float(np.percentile(valid_pixels, self.p_min))
        v_max = float(np.percentile(valid_pixels, self.p_max))

        if v_max - v_min < 1e-5:
            normalized = np.zeros_like(band, dtype=np.float32)
        else:
            normalized = np.clip((band.astype(np.float32) - v_min) / (v_max - v_min), 0.0, 1.0)

        if self.use_clahe:
            # Convert to uint8 for OpenCV CLAHE then back to float32
            u8 = (normalized * 255.0).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_grid_size, self.clahe_grid_size),
            )
            u8_clahe = clahe.apply(u8)
            normalized = u8_clahe.astype(np.float32) / 255.0

        if nodata_mask is not None:
            normalized[nodata_mask] = 0.0

        return normalized.astype(np.float32)

    def normalize_stack(
        self,
        stack: np.ndarray,
        nodata_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Normalizes multi-channel stack of shape (C, H, W).
        Returns float32 normalized stack with values in [0.0, 1.0].
        """
        c, h, w = stack.shape
        out = np.zeros((c, h, w), dtype=np.float32)
        for i in range(c):
            out[i] = self.normalize_band(stack[i], nodata_mask=nodata_mask)
        return out
