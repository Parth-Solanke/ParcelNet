from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from backend.app.core.logging import logger


class RasterReprojector:
    """Reprojects rasters to project CRS and aligns secondary grids to master ORI grid."""

    def __init__(self, target_crs_epsg: int = 32643):
        self.target_crs = CRS.from_epsg(target_crs_epsg)

    def reproject_to_target_crs(
        self,
        src_path: Path,
        dst_path: Path,
        resampling: Resampling = Resampling.bilinear,
    ) -> Path:
        """Reprojects src_path raster to target CRS."""
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(src_path) as src:
            if src.crs == self.target_crs:
                logger.info(f"{src_path.name} already in target CRS {self.target_crs}. Copying/passing through.")
                # If already in target CRS, write a clean copy
                profile = src.profile.copy()
                with rasterio.open(dst_path, "w", **profile) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
                return dst_path

            transform, width, height = calculate_default_transform(
                src.crs, self.target_crs, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update({
                "crs": self.target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            })

            with rasterio.open(dst_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.target_crs,
                        resampling=resampling,
                    )
        logger.info(f"Reprojected {src_path.name} to {dst_path.name} (EPSG:{self.target_crs.to_epsg()})")
        return dst_path

    def align_to_master(
        self,
        slave_path: Path,
        master_path: Path,
        dst_path: Path,
        resampling: Resampling = Resampling.bilinear,
    ) -> Path:
        """Resamples and clips/aligns slave raster directly to the master raster's pixel grid."""
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(master_path) as master:
            target_crs = master.crs
            target_transform = master.transform
            target_width = master.width
            target_height = master.height

        with rasterio.open(slave_path) as slave:
            profile = slave.profile.copy()
            profile.update({
                "crs": target_crs,
                "transform": target_transform,
                "width": target_width,
                "height": target_height,
            })

            with rasterio.open(dst_path, "w", **profile) as dst:
                for i in range(1, slave.count + 1):
                    destination = np.zeros((target_height, target_width), dtype=slave.dtypes[i - 1])
                    reproject(
                        source=rasterio.band(slave, i),
                        destination=destination,
                        src_transform=slave.transform,
                        src_crs=slave.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                    )
                    dst.write(destination, i)

        logger.info(f"Aligned {slave_path.name} to master grid of {master_path.name}")
        return dst_path
