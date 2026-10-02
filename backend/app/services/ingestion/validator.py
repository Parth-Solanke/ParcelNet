from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
from backend.app.core.logging import logger


class RasterValidationResult:
    def __init__(
        self,
        path: Path,
        raster_type: str,
        is_valid: bool,
        crs: Optional[str],
        crs_epsg: Optional[int],
        bounds: Tuple[float, float, float, float],
        width: int,
        height: int,
        count: int,
        dtype: str,
        nodata: Optional[float],
        gsd_x: float,
        gsd_y: float,
        errors: List[str],
        warnings: List[str],
    ):
        self.path = path
        self.raster_type = raster_type
        self.is_valid = is_valid
        self.crs = crs
        self.crs_epsg = crs_epsg
        self.bounds = bounds
        self.width = width
        self.height = height
        self.count = count
        self.dtype = dtype
        self.nodata = nodata
        self.gsd_x = gsd_x
        self.gsd_y = gsd_y
        self.errors = errors
        self.warnings = warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "type": self.raster_type,
            "is_valid": self.is_valid,
            "crs": self.crs,
            "crs_epsg": self.crs_epsg,
            "bounds": list(self.bounds),
            "width": self.width,
            "height": self.height,
            "bands": self.count,
            "dtype": self.dtype,
            "nodata": self.nodata,
            "gsd": round((abs(self.gsd_x) + abs(self.gsd_y)) / 2.0, 4),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class RasterValidator:
    """Validates CRS, resolution, nodata, and band count of ingested GeoTIFFs."""

    def __init__(self, target_crs_epsg: int = 32643, max_gsd_m: float = 0.20):
        self.target_crs_epsg = target_crs_epsg
        self.max_gsd_m = max_gsd_m

    def validate_file(self, file_path: Path, raster_type: str) -> RasterValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if not file_path.exists():
            return RasterValidationResult(
                path=file_path,
                raster_type=raster_type,
                is_valid=False,
                crs=None,
                crs_epsg=None,
                bounds=(0, 0, 0, 0),
                width=0,
                height=0,
                count=0,
                dtype="unknown",
                nodata=None,
                gsd_x=0.0,
                gsd_y=0.0,
                errors=[f"File does not exist: {file_path}"],
                warnings=[],
            )

        try:
            with rasterio.open(file_path) as src:
                crs = str(src.crs) if src.crs else None
                epsg = src.crs.to_epsg() if src.crs else None

                if not src.crs:
                    errors.append(f"{raster_type} has no defined CRS.")
                elif epsg and epsg != self.target_crs_epsg:
                    warnings.append(
                        f"{raster_type} CRS EPSG:{epsg} differs from target EPSG:{self.target_crs_epsg}. Reprojection required."
                    )

                res_x, res_y = abs(src.res[0]), abs(src.res[1])
                avg_gsd = (res_x + res_y) / 2.0

                if avg_gsd > self.max_gsd_m and (src.crs and src.crs.is_projected):
                    warnings.append(
                        f"GSD of {avg_gsd:.3f}m exceeds recommended max threshold of {self.max_gsd_m}m."
                    )

                if raster_type == "ORI" and src.count < 3:
                    errors.append(f"ORI requires at least 3 bands (RGB), found {src.count}.")
                elif raster_type in ["DSM", "DTM", "NDSM"] and src.count < 1:
                    errors.append(f"{raster_type} requires at least 1 elevation band, found {src.count}.")

                bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

                return RasterValidationResult(
                    path=file_path,
                    raster_type=raster_type,
                    is_valid=len(errors) == 0,
                    crs=crs,
                    crs_epsg=epsg,
                    bounds=bounds,
                    width=src.width,
                    height=src.height,
                    count=src.count,
                    dtype=str(src.dtypes[0]),
                    nodata=src.nodata,
                    gsd_x=res_x,
                    gsd_y=res_y,
                    errors=errors,
                    warnings=warnings,
                )
        except Exception as e:
            logger.error(f"Error inspecting {file_path}: {e}")
            return RasterValidationResult(
                path=file_path,
                raster_type=raster_type,
                is_valid=False,
                crs=None,
                crs_epsg=None,
                bounds=(0, 0, 0, 0),
                width=0,
                height=0,
                count=0,
                dtype="unknown",
                nodata=None,
                gsd_x=0.0,
                gsd_y=0.0,
                errors=[f"Failed to read raster: {str(e)}"],
                warnings=[],
            )

    @staticmethod
    def compute_overlap_ratio(
        bounds_a: Tuple[float, float, float, float],
        bounds_b: Tuple[float, float, float, float],
    ) -> float:
        """Computes intersection over union (IoU) of bounding boxes."""
        box_a = box(*bounds_a)
        box_b = box(*bounds_b)
        if not box_a.intersects(box_b):
            return 0.0
        inter_area = box_a.intersection(box_b).area
        min_area = min(box_a.area, box_b.area)
        if min_area <= 0:
            return 0.0
        return float(inter_area / min_area)
