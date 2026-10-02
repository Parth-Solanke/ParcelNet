from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import geopandas as gpd
import torch

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.ingestion.pipeline import IngestionPipeline
from backend.app.services.landuse.classifier import ObjectBasedLandUseRefiner
from backend.app.services.preprocessing.stacker import FeatureStackBuilder
from backend.app.services.topology.line_network import BoundaryLineNetworkBuilder
from backend.app.services.topology.polygonizer import FacePolygonizer
from backend.app.services.validation.auto_fixer import AutoFixEngine
from backend.app.services.validation.validator import CadastraValidator
from backend.app.services.vectorization.building_vectorizer import BuildingVectorizer
from backend.app.services.vectorization.road_vectorizer import RoadVectorizer
from ml.infer.tiled_infer import TiledPredictor
from ml.train.model import MultiTaskCadastraNet


class PipelineOrchestrator:
    """
    Executes the complete CadastraAI GeoAI pipeline:
    Ingest -> Preprocess -> Infer -> Vectorize -> Topology -> Landuse -> Validate -> Auto-fix.
    """

    def __init__(
        self,
        project_id: str,
        work_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.project_id = project_id
        self.work_dir = work_dir or (settings.data_dir / "projects" / project_id)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback or (lambda p, m: logger.info(f"[{p*100:.1f}%] {m}"))

    def run_pipeline(
        self,
        ori_path: Path,
        dsm_path: Optional[Path] = None,
        dtm_path: Optional[Path] = None,
        existing_parcels_path: Optional[Path] = None,
        model_weights_path: Optional[Path] = None,
        auto_heal: bool = True,
    ) -> Dict[str, Any]:
        """Runs the complete autonomous urban cadastral mapping pipeline."""
        results: Dict[str, Any] = {}

        # Step 1: Ingestion and Georeferencing QC (10%)
        self.progress_callback(0.10, "Step 1/8: Ingestion and Georeferencing QC")
        ingest_dir = self.work_dir / "ingest"
        ingest_pipeline = IngestionPipeline(output_dir=ingest_dir)
        ingest_res = ingest_pipeline.process(
            ori_path=ori_path,
            dsm_path=dsm_path,
            dtm_path=dtm_path,
            vector_path=existing_parcels_path,
        )
        results["ingestion_qc"] = ingest_res["qc_report"]
        ori_cog = Path(ingest_res["processed_rasters"]["ORI"])
        dsm_cog = Path(ingest_res["processed_rasters"].get("DSM", "")) if "DSM" in ingest_res["processed_rasters"] else None
        dtm_cog = Path(ingest_res["processed_rasters"].get("DTM", "")) if "DTM" in ingest_res["processed_rasters"] else None

        # Step 2: Pre-processing & Feature Stacking (25%)
        self.progress_callback(0.25, "Step 2/8: Pre-processing & Feature Stacking")
        stack_path = self.work_dir / "stack_4ch.tif"
        stack_builder = FeatureStackBuilder(channel_preset="rgb+ndsm")
        stack_builder.build_stack(
            ori_path=ori_cog,
            dsm_path=dsm_cog,
            dtm_path=dtm_cog,
            output_stack_path=stack_path,
        )
        results["stack_path"] = str(stack_path)

        # Step 3: Multi-Task Tiled Inference (45%)
        self.progress_callback(0.45, "Step 3/8: Multi-Task AI Tiled Inference")
        model = MultiTaskCadastraNet(in_channels=4, pretrained=False)
        if model_weights_path and model_weights_path.exists():
            checkpoint = torch.load(model_weights_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])

        predictor = TiledPredictor(model=model, tile_size_px=256, overlap_ratio=0.25, use_tta=False)
        pred_dir = self.work_dir / "predictions"
        pred_paths = predictor.predict_raster(stack_path, pred_dir)
        results["predictions"] = {k: str(v) for k, v in pred_paths.items()}

        # Step 4: Vectorize Features (60%)
        self.progress_callback(0.60, "Step 4/8: Vectorizing Buildings and Road Centerlines")
        vectors_dir = self.work_dir / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)

        b_vec = BuildingVectorizer()
        bldgs_gdf = b_vec.vectorize(pred_paths["buildings"], ndsm_path=dsm_cog)
        bldgs_path = vectors_dir / "buildings.geojson"
        bldgs_gdf.to_file(bldgs_path, driver="GeoJSON")

        r_vec = RoadVectorizer()
        c_lines_gdf, r_polys_gdf, _ = r_vec.vectorize(pred_paths["roads"])
        c_lines_path = vectors_dir / "roads_centerlines.geojson"
        r_polys_path = vectors_dir / "roads_polygons.geojson"
        c_lines_gdf.to_file(c_lines_path, driver="GeoJSON")
        r_polys_gdf.to_file(r_polys_path, driver="GeoJSON")

        results["buildings_count"] = len(bldgs_gdf)
        results["roads_count"] = len(c_lines_gdf)

        # Step 5: Topology Generation (75%)
        self.progress_callback(0.75, "Step 5/8: Generating Parcel Topology")
        line_builder = BoundaryLineNetworkBuilder()
        noded_lines = line_builder.build_network(
            boundaries_prob_path=pred_paths["boundaries"],
            road_polygons=r_polys_gdf,
            building_polygons=bldgs_gdf,
        )

        polygonizer = FacePolygonizer()
        parcels_gdf = polygonizer.polygonize_network(
            noded_lines=noded_lines,
            road_polygons=r_polys_gdf,
            building_polygons=bldgs_gdf,
            crs=bldgs_gdf.crs,
        )

        # If zero parcels from AI boundary map, incorporate existing parcels as fallback baseline
        if len(parcels_gdf) == 0 and existing_parcels_path and existing_parcels_path.exists():
            parcels_gdf = gpd.read_file(existing_parcels_path)
            parcels_gdf["confidence"] = 0.85
            parcels_gdf["source"] = "existing"
            parcels_gdf["status"] = "auto"

        # Step 6: Land-Use Classification (85%)
        self.progress_callback(0.85, "Step 6/8: Refining Parcel Land-Use")
        lu_refiner = ObjectBasedLandUseRefiner()
        classified_parcels = lu_refiner.refine_parcels(
            parcels_gdf=parcels_gdf,
            landuse_raster_path=pred_paths["landuse"],
            ndsm_raster_path=dsm_cog,
        )

        # Step 7: Validation & Auto-Healing (95%)
        self.progress_callback(0.95, "Step 7/8: Automated Topology & Geometry Validation")
        val_dir = self.work_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        validator = CadastraValidator()
        pre_val = validator.validate(classified_parcels)

        final_parcels = classified_parcels
        if auto_heal:
            fixer = AutoFixEngine()
            final_parcels, audit = fixer.auto_fix_all(classified_parcels)
            results["auto_fix_actions"] = len(audit)

        post_val = validator.validate(final_parcels)
        results["validation"] = post_val["summary"]

        parcels_out = vectors_dir / "parcels_final.geojson"
        final_parcels.to_file(parcels_out, driver="GeoJSON")
        results["final_parcels_path"] = str(parcels_out)
        results["parcels_count"] = len(final_parcels)

        # Step 8: Ready (100%)
        self.progress_callback(1.0, "Step 8/8: Pipeline completed successfully!")
        return results
