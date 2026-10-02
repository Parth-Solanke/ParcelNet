from pathlib import Path
from typing import List, Optional
import rasterio
from rasterio.enums import Resampling
from backend.app.core.logging import logger


class COGConverter:
    """Converts GeoTIFFs to Cloud-Optimized GeoTIFF (COG) layout with internal tiling and overviews."""

    def __init__(
        self,
        block_size: int = 512,
        compression: str = "DEFLATE",
        overview_factors: Optional[List[int]] = None,
    ):
        self.block_size = block_size
        self.compression = compression
        self.overview_factors = overview_factors or [2, 4, 8, 16]

    def convert(self, src_path: Path, dst_path: Path) -> Path:
        """Converts src_path to a tiled COG at dst_path."""
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": self.block_size,
                "blockysize": self.block_size,
                "compress": self.compression,
            })

            # Create tiled output
            with rasterio.open(dst_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    data = src.read(i)
                    dst.write(data, i)

                # Build pyramid overviews if raster is sufficiently large
                if src.width > self.block_size or src.height > self.block_size:
                    dst.build_overviews(self.overview_factors, Resampling.bilinear)
                    dst.update_tags(ns="rio_overview", resampling="bilinear")

        logger.info(f"Generated Cloud-Optimized GeoTIFF: {dst_path}")
        return dst_path
