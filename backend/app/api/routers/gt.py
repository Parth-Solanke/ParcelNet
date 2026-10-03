from pathlib import Path
from typing import Any, Dict
import math, shutil
import geopandas as gpd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.services.gt_integration.importer import GTImporter
from backend.app.services.gt_integration.matcher import GTMatcher
from backend.app.services.gt_integration.verification_list import FieldVerificationExporter

router = APIRouter(prefix="/projects/{project_id}/gt", tags=["gt"])


def _clean_nan(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON safety."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(i) for i in obj]
    return obj


def _safe_json_response(data: Any) -> JSONResponse:
    """Returns a JSONResponse with NaN-safe content."""
    return JSONResponse(content=_clean_nan(data))


def _resolve_parcels_path():
    """Returns the best available parcels file path."""
    candidates = [
        settings.data_dir / "processed" / "validation" / "parcels_healed.geojson",
        settings.data_dir / "processed" / "vectors" / "parcels_landuse.geojson",
        settings.data_dir / "raw" / "synthetic_parcels.geojson",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise HTTPException(status_code=404, detail="No parcels file found. Run the pipeline first.")


@router.post("/import")
async def import_gt_points(project_id: str, file: UploadFile = File(...)):
    gt_dir = settings.data_dir / "processed" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    saved_path = gt_dir / file.filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    importer = GTImporter()
    if saved_path.suffix.lower() == ".csv":
        gt_gdf = importer.import_points_csv(saved_path)
    else:
        gt_gdf = importer.import_vector(saved_path)

    parcels_path = _resolve_parcels_path()
    parcels_gdf = gpd.read_file(parcels_path)
    matcher = GTMatcher()
    verified_gdf, metrics = matcher.match_points_to_parcels(parcels_gdf, gt_gdf)

    # Write to healed path (not overwrite raw synthetic data)
    out_path = settings.data_dir / "processed" / "validation" / "parcels_healed.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verified_gdf.to_file(out_path, driver="GeoJSON")

    return _safe_json_response({
        "status": "success",
        "imported_points": len(gt_gdf),
        "metrics": metrics,
    })


@router.get("/verification-list")
async def get_field_verification_list(project_id: str):
    parcels_path = _resolve_parcels_path()
    parcels_gdf = gpd.read_file(parcels_path)

    # Ensure WGS84 for field apps (QField / Mergin Maps)
    if parcels_gdf.crs is None:
        parcels_gdf = parcels_gdf.set_crs(epsg=32643)
    if parcels_gdf.crs.to_epsg() != 4326:
        parcels_gdf = parcels_gdf.to_crs(epsg=4326)

    exporter = FieldVerificationExporter()
    v_gdf = exporter.create_verification_list(parcels_gdf)

    # Fallback: if no parcels need verification, return all parcels for surveyor review
    if len(v_gdf) == 0:
        v_gdf = parcels_gdf.copy()
        v_gdf["surveyor_name"] = ""
        v_gdf["field_verdict"] = "pending"
        v_gdf["boundary_notes"] = ""
        v_gdf["inspection_date"] = ""

    return _safe_json_response(v_gdf.__geo_interface__)
