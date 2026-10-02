from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.dtos import QCReportDTO
from backend.app.services.ingestion.cog_converter import COGConverter
from backend.app.services.ingestion.qc_reporter import QCReporter
from backend.app.services.ingestion.reprojector import RasterReprojector
from backend.app.services.ingestion.validator import RasterValidationResult, RasterValidator
from backend.app.services.ingestion.vector_ingest import VectorIngestor, VectorIngestResult


class IngestionPipeline:
    """End-to-end ingestion pipeline executing validation, reprojection, COG generation, and QC."""

    def __init__(self, target_crs_epsg: Optional[int] = None, output_dir: Optional[Path] = None):
        self.target_crs_epsg = target_crs_epsg or settings.project.default_crs_epsg
        self.output_dir = output_dir or (settings.data_dir / "processed")
        self.validator = RasterValidator(target_crs_epsg=self.target_crs_epsg)
        self.reprojector = RasterReprojector(target_crs_epsg=self.target_crs_epsg)
        self.cog_converter = COGConverter()
        self.vector_ingestor = VectorIngestor(target_crs_epsg=self.target_crs_epsg)
        self.qc_reporter = QCReporter(target_crs_epsg=self.target_crs_epsg)

    def process(
        self,
        ori_path: Path,
        dsm_path: Optional[Path] = None,
        dtm_path: Optional[Path] = None,
        vector_path: Optional[Path] = None,
        control_points: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Runs complete ingestion and georeferencing QC."""
        logger.info(f"Starting ingestion for ORI: {ori_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        raster_paths = {"ORI": ori_path}
        if dsm_path:
            raster_paths["DSM"] = dsm_path
        if dtm_path:
            raster_paths["DTM"] = dtm_path

        # Step 1: Initial Validation
        validation_results: Dict[str, RasterValidationResult] = {}
        for r_type, path in raster_paths.items():
            validation_results[r_type] = self.validator.validate_file(path, r_type)

        # Step 2: Reproject and align rasters
        processed_rasters: Dict[str, Path] = {}
        # Master ORI reprojection & COG conversion
        reproj_ori = self.output_dir / "ori_reproj.tif"
        self.reprojector.reproject_to_target_crs(ori_path, reproj_ori)
        cog_ori = self.output_dir / "ori_cog.tif"
        self.cog_converter.convert(reproj_ori, cog_ori)
        processed_rasters["ORI"] = cog_ori

        # Align DSM to Master ORI
        if dsm_path and dsm_path.exists():
            reproj_dsm = self.output_dir / "dsm_aligned.tif"
            self.reprojector.align_to_master(dsm_path, cog_ori, reproj_dsm)
            cog_dsm = self.output_dir / "dsm_cog.tif"
            self.cog_converter.convert(reproj_dsm, cog_dsm)
            processed_rasters["DSM"] = cog_dsm

        # Align DTM to Master ORI
        if dtm_path and dtm_path.exists():
            reproj_dtm = self.output_dir / "dtm_aligned.tif"
            self.reprojector.align_to_master(dtm_path, cog_ori, reproj_dtm)
            cog_dtm = self.output_dir / "dtm_cog.tif"
            self.cog_converter.convert(reproj_dtm, cog_dtm)
            processed_rasters["DTM"] = cog_dtm

        # Step 3: Vector ingestion if provided
        vector_result: Optional[VectorIngestResult] = None
        if vector_path and vector_path.exists():
            cleaned_vector_path = self.output_dir / "existing_parcels.geojson"
            _, vector_result = self.vector_ingestor.ingest_vector_layer(
                vector_path, output_cleaned_path=cleaned_vector_path
            )

        # Step 4: Re-evaluate post-alignment for final QC Report
        final_validation: Dict[str, RasterValidationResult] = {}
        for r_type, path in processed_rasters.items():
            final_validation[r_type] = self.validator.validate_file(path, r_type)

        qc_report = self.qc_reporter.evaluate_pipeline(
            raster_results=final_validation,
            vector_results={"existing_parcels": vector_result} if vector_result else None,
            control_points=control_points,
        )
        report_path = self.output_dir / "ingestion_qc_report.json"
        self.qc_reporter.save_report(qc_report, report_path)

        return {
            "processed_rasters": {k: str(v) for k, v in processed_rasters.items()},
            "qc_report": qc_report.model_dump(),
            "report_path": str(report_path),
        }
