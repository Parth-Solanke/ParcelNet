from pathlib import Path
from typing import Any, Dict, List, Optional
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from backend.app.core.logging import logger
from ml.datasets.label_rasterizer import LabelRasterizer


class ObjectBasedLandUseRefiner:
    """
    Performs object-based land-use refinement aggregating pixel predictions
    within parcel polygons and applying physical rule-based elevation checks.
    """

    CLASS_NAMES = {
        0: "background",
        1: "residential",
        2: "commercial",
        3: "institutional",
        4: "open_land",
        5: "vegetation",
        6: "water",
        7: "road",
        8: "built_up_other",
    }

    def __init__(self, tall_building_height_m: float = 3.5):
        self.tall_threshold = tall_building_height_m

    def refine_parcels(
        self,
        parcels_gdf: gpd.GeoDataFrame,
        landuse_raster_path: Path,
        ndsm_raster_path: Optional[Path] = None,
    ) -> gpd.GeoDataFrame:
        """
        Classifies each parcel polygon by majority pixel vote with elevation sanity checks.
        """
        if len(parcels_gdf) == 0:
            return parcels_gdf

        refined_gdf = parcels_gdf.copy()
        classes = []
        confidences = []

        with rasterio.open(landuse_raster_path) as lu_src:
            ndsm_src = rasterio.open(ndsm_raster_path) if ndsm_raster_path and ndsm_raster_path.exists() else None

            for idx, row in refined_gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty or not geom.is_valid:
                    classes.append("open_land")
                    confidences.append(0.50)
                    continue

                try:
                    # Extract pixels within polygon
                    masked_lu, _ = mask(lu_src, [geom], crop=True, nodata=255)
                    valid_pixels = masked_lu[0][masked_lu[0] != 255].flatten()

                    if len(valid_pixels) == 0:
                        # Fallback
                        classes.append("open_land")
                        confidences.append(0.50)
                        continue

                    # Majority voting
                    counts = np.bincount(valid_pixels.astype(np.int64))
                    majority_class_id = int(np.argmax(counts))
                    majority_ratio = float(counts[majority_class_id] / len(valid_pixels))
                    predicted_class = self.CLASS_NAMES.get(majority_class_id, "residential")

                    # Rule-based physical sanity check using nDSM elevation
                    if ndsm_src is not None:
                        masked_ndsm, _ = mask(ndsm_src, [geom], crop=True, nodata=-9999.0)
                        ndsm_pixels = masked_ndsm[0][masked_ndsm[0] != -9999.0].flatten()
                        if len(ndsm_pixels) > 0:
                            median_height = float(np.median(ndsm_pixels))
                            # Tall structure cannot be pure vegetation or water
                            if median_height > self.tall_threshold and predicted_class in ["vegetation", "water", "open_land"]:
                                predicted_class = "residential"
                                majority_ratio = max(majority_ratio, 0.75)
                            # Completely flat cannot be commercial/institutional with heavy structures
                            elif median_height < 0.3 and predicted_class in ["commercial", "institutional"]:
                                predicted_class = "open_land"

                    classes.append(predicted_class)
                    confidences.append(round(majority_ratio, 4))

                except Exception as e:
                    logger.warning(f"Error classifying parcel {row.get('id', idx)}: {e}")
                    classes.append("residential")
                    confidences.append(0.50)

            if ndsm_src is not None:
                ndsm_src.close()

        refined_gdf["landuse"] = classes
        refined_gdf["lu_confidence"] = confidences

        logger.info(f"Refined land-use classes for {len(refined_gdf)} parcels")
        return refined_gdf
