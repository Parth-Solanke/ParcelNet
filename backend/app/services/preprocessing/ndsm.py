from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import rasterio
from scipy.ndimage import median_filter
from backend.app.core.config import settings
from backend.app.core.logging import logger


class NDSMCalculator:
    """Computes Normalized Digital Surface Model (nDSM = DSM - DTM) with noise filtration."""

    def __init__(
        self,
        min_height_threshold_m: float = 0.5,
        median_filter_size: int = 3,
    ):
        self.min_height_threshold_m = min_height_threshold_m
        self.median_filter_size = median_filter_size

    def compute_ndsm_array(
        self,
        dsm_data: np.ndarray,
        dtm_data: np.ndarray,
        nodata_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes nDSM array with negative clipping and noise filtering."""
        # Calculate raw difference
        ndsm = dsm_data.astype(np.float32) - dtm_data.astype(np.float32)

        # Clip negatives to 0 (ground level)
        ndsm = np.maximum(ndsm, 0.0)

        # Apply noise threshold (ground noise < min_height_threshold_m set to 0)
        ndsm[ndsm < self.min_height_threshold_m] = 0.0

        # Apply median filter to remove single-pixel spike artifacts
        if self.median_filter_size > 1:
            ndsm = median_filter(ndsm, size=self.median_filter_size)

        # Restore nodata mask if provided
        if nodata_mask is not None:
            ndsm[nodata_mask] = 0.0

        return ndsm

    def compute_from_files(
        self,
        dsm_path: Path,
        dtm_path: Path,
        output_path: Path,
    ) -> Path:
        """Reads aligned DSM and DTM files and writes processed nDSM GeoTIFF."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
            if (dsm_src.width != dtm_src.width) or (dsm_src.height != dtm_src.height):
                raise ValueError(
                    f"DSM ({dsm_src.width}x{dsm_src.height}) and DTM ({dtm_src.width}x{dtm_src.height}) "
                    "pixel dimensions do not match. Re-run alignment first."
                )

            profile = dsm_src.profile.copy()
            profile.update({
                "count": 1,
                "dtype": "float32",
                "nodata": -9999.0,
                "compress": "DEFLATE",
            })

            dsm = dsm_src.read(1)
            dtm = dtm_src.read(1)

            # Build nodata mask
            nodata_mask = np.zeros(dsm.shape, dtype=bool)
            if dsm_src.nodata is not None:
                nodata_mask |= (dsm == dsm_src.nodata)
            if dtm_src.nodata is not None:
                nodata_mask |= (dtm == dtm_src.nodata)

            ndsm = self.compute_ndsm_array(dsm, dtm, nodata_mask)
            ndsm[nodata_mask] = -9999.0

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(ndsm.astype(np.float32), 1)

        logger.info(f"Successfully computed nDSM: {output_path}")
        return output_path
