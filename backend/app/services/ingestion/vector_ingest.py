from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
from shapely.validation import make_valid
from backend.app.core.logging import logger


class VectorIngestResult:
    def __init__(
        self,
        path: Path,
        layer_type: str,
        feature_count: int,
        crs: str,
        bounds: List[float],
        repaired_geometries_count: int,
        dropped_empty_count: int,
    ):
        self.path = path
        self.layer_type = layer_type
        self.feature_count = feature_count
        self.crs = crs
        self.bounds = bounds
        self.repaired_geometries_count = repaired_geometries_count
        self.dropped_empty_count = dropped_empty_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "layer_type": self.layer_type,
            "feature_count": self.feature_count,
            "crs": self.crs,
            "bounds": self.bounds,
            "repaired_geometries": self.repaired_geometries_count,
            "dropped_empty": self.dropped_empty_count,
        }


class VectorIngestor:
    """Ingests vector layers, reprojects to project CRS, repairs invalid geometries, and drops empties."""

    def __init__(self, target_crs_epsg: int = 32643):
        self.target_crs_epsg = target_crs_epsg

    def ingest_vector_layer(
        self,
        file_path: Path,
        layer_type: str = "existing_parcels",
        output_cleaned_path: Optional[Path] = None,
    ) -> Tuple[gpd.GeoDataFrame, VectorIngestResult]:
        """Reads vector file, fixes CRS, runs make_valid, and drops empties."""
        gdf = gpd.read_file(file_path)
        original_count = len(gdf)

        # Reproject if needed
        if gdf.crs is None:
            logger.warning(f"Vector layer {file_path.name} has no CRS. Assuming EPSG:{self.target_crs_epsg}.")
            gdf.set_crs(epsg=self.target_crs_epsg, inplace=True)
        elif gdf.crs.to_epsg() != self.target_crs_epsg:
            logger.info(f"Reprojecting vector layer from {gdf.crs} to EPSG:{self.target_crs_epsg}")
            gdf = gdf.to_crs(epsg=self.target_crs_epsg)

        # Drop empty or null geometries
        non_empty_mask = ~gdf.geometry.is_empty & gdf.geometry.notnull()
        dropped_empty = int(np.sum(~non_empty_mask))
        gdf = gdf[non_empty_mask].copy()

        # Run make_valid on invalid geometries
        invalid_mask = ~gdf.geometry.is_valid
        repaired_count = int(np.sum(invalid_mask))
        if repaired_count > 0:
            logger.info(f"Fixing {repaired_count} invalid geometries in {file_path.name} using make_valid.")
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)

        # Keep valid geometries (polygons / multipolygons for parcels)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

        bounds = [float(b) for b in gdf.total_bounds] if len(gdf) > 0 else [0.0, 0.0, 0.0, 0.0]

        if output_cleaned_path:
            output_cleaned_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_cleaned_path, driver="GeoJSON")

        result = VectorIngestResult(
            path=file_path,
            layer_type=layer_type,
            feature_count=len(gdf),
            crs=f"EPSG:{self.target_crs_epsg}",
            bounds=bounds,
            repaired_geometries_count=repaired_count,
            dropped_empty_count=dropped_empty,
        )
        return gdf, result
