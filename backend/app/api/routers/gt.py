from pathlib import Path
from typing import Any, Dict
import geopandas as gpd
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.config import settings
from backend.app.services.gt_integration.importer import GTImporter
from backend.app.services.gt_integration.matcher import GTMatcher
from backend.app.services.gt_integration.verification_list import FieldVerificationExporter

router = APIRouter(prefix="/projects/{project_id}/gt", tags=["gt"])


@router.post("/import")
async def import_gt_points(project_id: str, file: UploadFile = File(...)):
    gt_dir = settings.data_dir / "processed" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    saved_path = gt_dir / file.filename

    with open(saved_path, "wb") as buffer:
        import shutil
        shutil.copyfileobj(file.file, buffer)

    importer = GTImporter()
    if saved_path.suffix.lower() == ".csv":
        gt_gdf = importer.import_points_csv(saved_path)
    else:
        gt_gdf = importer.import_vector(saved_path)

    # Match with current parcels
    parcels_path = settings.data_dir / "processed" / "validation" / "parcels_healed.geojson"
    if not parcels_path.exists():
        parcels_path = settings.data_dir / "raw" / "synthetic_parcels.geojson"

    parcels_gdf = gpd.read_file(parcels_path)
    matcher = GTMatcher()
    verified_gdf, metrics = matcher.match_points_to_parcels(parcels_gdf, gt_gdf)
    verified_gdf.to_file(parcels_path, driver="GeoJSON")

    return {
        "status": "success",
        "imported_points": len(gt_gdf),
        "metrics": metrics,
    }


@router.get("/verification-list")
async def get_field_verification_list(project_id: str):
    parcels_path = settings.data_dir / "processed" / "validation" / "parcels_healed.geojson"
    if not parcels_path.exists():
        parcels_path = settings.data_dir / "raw" / "synthetic_parcels.geojson"

    parcels_gdf = gpd.read_file(parcels_path)
    exporter = FieldVerificationExporter()
    v_gdf = exporter.create_verification_list(parcels_gdf)
    return v_gdf.__geo_interface__
