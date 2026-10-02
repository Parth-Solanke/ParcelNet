from pathlib import Path
from typing import Any, Dict, List, Optional
import geopandas as gpd
import numpy as np
from backend.app.core.logging import logger


class FieldVerificationExporter:
    """Generates field verification worklists for mobile GIS apps (QField / Mergin Maps)."""

    def __init__(self, confidence_threshold: float = 0.65):
        self.conf_threshold = confidence_threshold

    def create_verification_list(
        self,
        parcels_gdf: gpd.GeoDataFrame,
        issues_gdf: Optional[gpd.GeoDataFrame] = None,
        output_dir: Optional[Path] = None,
    ) -> gpd.GeoDataFrame:
        """
        Filters parcels flagged with errors, low confidence, or disputed boundaries.
        Prepares standardized survey schema for mobile surveyors.
        """
        mask = np.zeros(len(parcels_gdf), dtype=bool)
        if "confidence" in parcels_gdf.columns:
            mask |= (parcels_gdf["confidence"] < self.conf_threshold).to_numpy()
        if "needs_gt" in parcels_gdf.columns:
            mask |= (parcels_gdf["needs_gt"] == True).to_numpy()
        if "status" in parcels_gdf.columns:
            mask |= (parcels_gdf["status"] == "disputed").to_numpy()
        needs_verification_mask = mask

        verification_gdf = parcels_gdf[needs_verification_mask].copy()

        # Add surveyor checklist fields
        verification_gdf["surveyor_name"] = ""
        verification_gdf["field_verdict"] = "pending"  # approved, needs_shift, split, merge, rejected
        verification_gdf["boundary_notes"] = ""
        verification_gdf["inspection_date"] = ""

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            geojson_out = output_dir / "field_verification_list.geojson"
            csv_out = output_dir / "field_verification_list.csv"

            verification_gdf.to_file(geojson_out, driver="GeoJSON")
            verification_gdf.drop(columns=["geometry"]).to_csv(csv_out, index=False)
            logger.info(
                f"Exported field verification list ({len(verification_gdf)} parcels) -> {geojson_out}"
            )

        return verification_gdf
