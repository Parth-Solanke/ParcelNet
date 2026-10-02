from pathlib import Path
from typing import Any, Dict, List, Optional
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from backend.app.core.logging import logger


class GTImporter:
    """Imports Ground Truthing (GT) points and CORS control points from CSV or GeoJSON."""

    def __init__(self, target_crs_epsg: int = 32643):
        self.target_crs = f"EPSG:{target_crs_epsg}"

    def import_points_csv(
        self,
        csv_path: Path,
        x_col: str = "x",
        y_col: str = "y",
        source_crs: str = "EPSG:32643",
    ) -> gpd.GeoDataFrame:
        """Reads CSV file and converts coordinates to GeoDataFrame with reprojection."""
        df = pd.read_csv(csv_path)

        # Autodetect column names
        cols = {c.lower(): c for c in df.columns}
        col_x = cols.get(x_col.lower()) or cols.get("lon") or cols.get("longitude") or cols.get("easting")
        col_y = cols.get(y_col.lower()) or cols.get("lat") or cols.get("latitude") or cols.get("northing")

        if not col_x or not col_y:
            raise ValueError(f"Could not locate coordinate columns in CSV {csv_path.name}")

        geometry = [Point(xy) for xy in zip(df[col_x], df[col_y])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=source_crs)

        if gdf.crs.to_string() != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)

        logger.info(f"Imported {len(gdf)} GT points from {csv_path.name} in {self.target_crs}")
        return gdf

    def import_vector(self, file_path: Path) -> gpd.GeoDataFrame:
        """Reads vector layer (GeoJSON, GPKG, SHP) and reprojects to target CRS."""
        gdf = gpd.read_file(file_path)
        if gdf.crs is None:
            gdf.set_crs(self.target_crs, inplace=True)
        elif gdf.crs.to_string() != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)
        return gdf
