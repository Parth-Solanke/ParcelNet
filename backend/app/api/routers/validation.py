from pathlib import Path
from typing import Any, Dict, List
import geopandas as gpd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import json, math

from backend.app.api.routers.projects import PROJECTS_STORE
from backend.app.core.config import settings
from backend.app.services.validation.auto_fixer import AutoFixEngine
from backend.app.services.validation.validator import CadastraValidator

router = APIRouter(prefix="/projects/{project_id}/validation", tags=["validation"])


def _clean_nan(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(i) for i in obj]
    return obj


def _gdf_to_geojson(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Convert GeoDataFrame to JSON-safe GeoJSON dict."""
    if len(gdf) == 0:
        return {"type": "FeatureCollection", "features": []}
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=32643)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return _clean_nan(gdf.__geo_interface__)


@router.post("/run")
async def run_validation(project_id: str):
    parcels_path = settings.data_dir / "processed" / "vectors" / "parcels_landuse.geojson"
    if not parcels_path.exists():
        parcels_path = settings.data_dir / "raw" / "synthetic_parcels.geojson"
    if not parcels_path.exists():
        raise HTTPException(status_code=404, detail="No parcels file found to validate")

    gdf = gpd.read_file(parcels_path)
    validator = CadastraValidator()
    report = validator.validate(gdf)
    return _clean_nan(report["summary"])


@router.get("/issues")
async def get_validation_issues(project_id: str):
    issues_path = settings.data_dir / "processed" / "validation" / "validation_issues.geojson"
    if not issues_path.exists():
        return {"type": "FeatureCollection", "features": []}

    gdf = gpd.read_file(issues_path)
    return _gdf_to_geojson(gdf)


@router.post("/autofix")
async def apply_autofix(project_id: str):
    parcels_path = settings.data_dir / "processed" / "vectors" / "parcels_landuse.geojson"
    if not parcels_path.exists():
        parcels_path = settings.data_dir / "raw" / "synthetic_parcels.geojson"
    if not parcels_path.exists():
        raise HTTPException(status_code=404, detail="No parcels file found to heal")

    gdf = gpd.read_file(parcels_path)
    fixer = AutoFixEngine()
    healed_gdf, audit = fixer.auto_fix_all(gdf)

    out_dir = settings.data_dir / "processed" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parcels_healed.geojson"
    healed_gdf.to_file(out_path, driver="GeoJSON")

    validator = CadastraValidator()
    post_report = validator.validate(healed_gdf)

    return _clean_nan({
        "status": "success",
        "repairs_count": len(audit),
        "audit": audit,
        "post_validation_summary": post_report["summary"],
    })
