from pathlib import Path
from typing import List, Optional
import numpy as np
import rasterio
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.preprocessing.auxiliary import AuxiliaryFeatureGenerator
from backend.app.services.preprocessing.ndsm import NDSMCalculator
from backend.app.services.preprocessing.normalizer import RadiometricNormalizer


class FeatureStackBuilder:
    """Builds model-ready normalized feature stacks (RGB + nDSM + Auxiliary)."""

    def __init__(
        self,
        channel_preset: str = "rgb+ndsm",
        use_clahe: bool = False,
    ):
        self.channel_preset = channel_preset
        self.normalizer = RadiometricNormalizer(use_clahe=use_clahe)
        self.aux_gen = AuxiliaryFeatureGenerator()
        self.ndsm_calc = NDSMCalculator()

    def build_stack(
        self,
        ori_path: Path,
        dsm_path: Optional[Path],
        dtm_path: Optional[Path],
        output_stack_path: Path,
    ) -> Path:
        """
        Loads aligned rasters, computes nDSM and auxiliary channels, normalizes,
        and saves multi-band stack GeoTIFF.
        """
        output_stack_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(ori_path) as ori_src:
            profile = ori_src.profile.copy()
            h, w = ori_src.height, ori_src.width
            res_x, res_y = abs(ori_src.res[0]), abs(ori_src.res[1])

            # Read RGB (first 3 bands)
            r = ori_src.read(1)
            g = ori_src.read(2)
            b = ori_src.read(3)

            nodata_mask = np.zeros((h, w), dtype=bool)
            if ori_src.nodata is not None:
                nodata_mask |= (r == ori_src.nodata) | (g == ori_src.nodata) | (b == ori_src.nodata)

            # Normalize RGB
            r_norm = self.normalizer.normalize_band(r, nodata_mask)
            g_norm = self.normalizer.normalize_band(g, nodata_mask)
            b_norm = self.normalizer.normalize_band(b, nodata_mask)

            bands_to_stack: List[np.ndarray] = [r_norm, g_norm, b_norm]
            band_descriptions: List[str] = ["Red (normalized)", "Green (normalized)", "Blue (normalized)"]

            # Compute and append nDSM if required and inputs are available
            if "ndsm" in self.channel_preset:
                if dsm_path and dtm_path and dsm_path.exists() and dtm_path.exists():
                    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
                        dsm = dsm_src.read(1)
                        dtm = dtm_src.read(1)
                        ndsm = self.ndsm_calc.compute_ndsm_array(dsm, dtm, nodata_mask)
                        ndsm_norm = self.normalizer.normalize_band(ndsm, nodata_mask)
                        bands_to_stack.append(ndsm_norm)
                        band_descriptions.append("nDSM (normalized)")
                else:
                    logger.warning("nDSM requested but DSM or DTM not provided. Padding 4th channel with zeros.")
                    bands_to_stack.append(np.zeros((h, w), dtype=np.float32))
                    band_descriptions.append("nDSM (empty/padded)")

            # Compute and append Hillshade if requested
            if "hillshade" in self.channel_preset:
                if dsm_path and dsm_path.exists():
                    with rasterio.open(dsm_path) as dsm_src:
                        dsm = dsm_src.read(1)
                        hs = self.aux_gen.compute_hillshade(dsm, dx=res_x, dy=res_y)
                        hs_norm = self.normalizer.normalize_band(hs, nodata_mask)
                        bands_to_stack.append(hs_norm)
                        band_descriptions.append("Hillshade (normalized)")

            total_bands = len(bands_to_stack)
            profile.update({
                "count": total_bands,
                "dtype": "float32",
                "nodata": 0.0,
                "compress": "DEFLATE",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            })

            with rasterio.open(output_stack_path, "w", **profile) as dst:
                for idx, band_arr in enumerate(bands_to_stack, start=1):
                    dst.write(band_arr.astype(np.float32), idx)
                    dst.set_band_description(idx, band_descriptions[idx - 1])

        logger.info(f"Built {total_bands}-channel feature stack saved to: {output_stack_path}")
        return output_stack_path
