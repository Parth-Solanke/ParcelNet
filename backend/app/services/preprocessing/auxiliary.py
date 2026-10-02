from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import rasterio
from backend.app.core.logging import logger


class AuxiliaryFeatureGenerator:
    """Generates auxiliary spatial features: Hillshade, Slope, and Excess Green index."""

    def __init__(
        self,
        hillshade_azimuth_deg: float = 315.0,
        hillshade_altitude_deg: float = 45.0,
    ):
        self.azimuth_rad = np.radians(360.0 - hillshade_azimuth_deg + 90.0)
        self.altitude_rad = np.radians(hillshade_altitude_deg)

    def compute_hillshade(
        self,
        elevation: np.ndarray,
        dx: float = 0.05,
        dy: float = 0.05,
    ) -> np.ndarray:
        """Computes shaded relief (hillshade) from elevation array [0, 255]."""
        # Compute spatial gradients
        gy, gx = np.gradient(elevation, dy, dx)

        slope = np.pi / 2.0 - np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gx, gy)

        shaded = (
            np.sin(self.altitude_rad) * np.sin(slope)
            + np.cos(self.altitude_rad) * np.cos(slope) * np.cos(self.azimuth_rad - aspect)
        )
        # Rescale to 0 - 255
        hillshade = np.clip(255.0 * np.maximum(shaded, 0.0), 0, 255).astype(np.uint8)
        return hillshade

    def compute_slope_degrees(
        self,
        elevation: np.ndarray,
        dx: float = 0.05,
        dy: float = 0.05,
    ) -> np.ndarray:
        """Computes terrain slope in degrees [0, 90]."""
        gy, gx = np.gradient(elevation, dy, dx)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        return np.degrees(slope_rad).astype(np.float32)

    @staticmethod
    def compute_excess_green(
        rgb_data: np.ndarray,
    ) -> np.ndarray:
        """
        Computes Excess Green Index (ExG = 2G - R - B) proxy for vegetation.
        rgb_data shape: (3, H, W) where band 0=R, 1=G, 2=B.
        Returns normalized float32 array in range [0.0, 1.0].
        """
        r = rgb_data[0].astype(np.float32)
        g = rgb_data[1].astype(np.float32)
        b = rgb_data[2].astype(np.float32)

        # Standard Excess Green index
        exg = 2.0 * g - r - b

        # Normalize to 0.0 - 1.0
        min_val = np.percentile(exg, 2.0)
        max_val = np.percentile(exg, 98.0)
        if max_val - min_val > 1e-5:
            exg_norm = np.clip((exg - min_val) / (max_val - min_val), 0.0, 1.0)
        else:
            exg_norm = np.zeros_like(exg, dtype=np.float32)

        return exg_norm.astype(np.float32)
