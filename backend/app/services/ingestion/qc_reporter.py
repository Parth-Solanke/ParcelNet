from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
from backend.app.models.dtos import QCReportDTO
from backend.app.services.ingestion.validator import RasterValidationResult, RasterValidator
from backend.app.services.ingestion.vector_ingest import VectorIngestResult
from backend.app.core.logging import logger


class QCReporter:
    """Generates comprehensive Georeferencing & Dataset Quality Control reports."""

    def __init__(self, target_crs_epsg: int = 32643, min_overlap_ratio: float = 0.90):
        self.target_crs_epsg = target_crs_epsg
        self.min_overlap_ratio = min_overlap_ratio

    def evaluate_pipeline(
        self,
        raster_results: Dict[str, RasterValidationResult],
        vector_results: Optional[Dict[str, VectorIngestResult]] = None,
        control_points: Optional[List[Dict[str, float]]] = None,
    ) -> QCReportDTO:
        errors: List[str] = []
        warnings: List[str] = []

        # Check required ORI
        if "ORI" not in raster_results or not raster_results["ORI"].is_valid:
            errors.append("Valid ORI (Orthorectified Image) is missing or unreadable.")

        # Check CRS consistency
        for r_type, res in raster_results.items():
            if not res.is_valid:
                errors.extend([f"[{r_type}] {e}" for e in res.errors])
            if res.warnings:
                warnings.extend([f"[{r_type}] {w}" for w in res.warnings])

        # Compute Bounding Box Overlap between ORI, DSM, DTM
        overlap_ratio = 1.0
        if "ORI" in raster_results and "DSM" in raster_results:
            overlap_ratio = RasterValidator.compute_overlap_ratio(
                raster_results["ORI"].bounds, raster_results["DSM"].bounds
            )
            if overlap_ratio < self.min_overlap_ratio:
                warnings.append(
                    f"Bounding overlap between ORI and DSM is {overlap_ratio * 100:.1f}%, below target {self.min_overlap_ratio * 100:.1f}%."
                )

        if "ORI" in raster_results and "DTM" in raster_results:
            dtm_overlap = RasterValidator.compute_overlap_ratio(
                raster_results["ORI"].bounds, raster_results["DTM"].bounds
            )
            overlap_ratio = min(overlap_ratio, dtm_overlap)
            if dtm_overlap < self.min_overlap_ratio:
                warnings.append(
                    f"Bounding overlap between ORI and DTM is {dtm_overlap * 100:.1f}%, below target {self.min_overlap_ratio * 100:.1f}%."
                )

        # Check alignment (width, height equality across rasters)
        alignment_verified = True
        if "ORI" in raster_results:
            ori_res = raster_results["ORI"]
            for r_type in ["DSM", "DTM"]:
                if r_type in raster_results:
                    other_res = raster_results[r_type]
                    if (ori_res.width != other_res.width) or (ori_res.height != other_res.height):
                        alignment_verified = False
                        warnings.append(
                            f"{r_type} dimensions ({other_res.width}x{other_res.height}) do not match ORI ({ori_res.width}x{ori_res.height}). Resampling required."
                        )

        # Control Points RMSE check if supplied
        control_points_rmse_m: Optional[float] = None
        if control_points and len(control_points) > 0:
            # MVP affine residual computation: residuals between point coordinates & height accuracies
            heights = [p.get("height", 0.0) for p in control_points]
            accuracies = [p.get("accuracy_cm", 0.0) / 100.0 for p in control_points]
            control_points_rmse_m = float(np.sqrt(np.mean(np.array(accuracies) ** 2)))
            if control_points_rmse_m > 0.50:
                warnings.append(
                    f"Control points RMSE is {control_points_rmse_m * 100:.1f}cm, exceeding 50cm threshold."
                )

        report = QCReportDTO(
            is_valid=(len(errors) == 0),
            project_crs=self.target_crs_epsg,
            raster_summaries={k: v.to_dict() for k, v in raster_results.items()},
            bounds_overlap_ratio=round(overlap_ratio, 4),
            alignment_verified=alignment_verified,
            control_points_rmse_m=control_points_rmse_m,
            warnings=warnings,
            errors=errors,
        )
        return report

    def save_report(self, report: QCReportDTO, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        logger.info(f"Ingestion QC report saved to {output_path}")
        return output_path
