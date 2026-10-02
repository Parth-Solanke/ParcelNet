from pathlib import Path
from typing import Any, Dict, List
import geopandas as gpd
from fastapi import APIRouter, HTTPException

from backend.app.api.routers.projects import PROJECTS_STORE
from backend.app.core.config import settings
from backend.app.services.validation.auto_fixer import AutoFixEngine
from backend.app.services.validation.validator import CadastraValidator

router = APIRouter(prefix="/projects/{project_id}/validation", tags=["validation"])


@router.post("/run")
async def run_validation(project_id: str):
    parcels_path = settings.data_dir / "processed" / "vectors" / "parcels_landuse.geojson"
    if not parcels_path.exists():
        parcels_path = settings.data_dir / "raw" / "synthetic_parcels.geojson"

    gdf = gpd.read_file(parcels_path)
    validator = CadastraValidator()
    report = validator.validate(gdf)
    return report["summary"]


@router.get("/issues")
async def get_validation_issues(project_id: str):
    issues_path = settings.data_dir / "processed" / "validation" / "validation_issues.geojson"
    if not issues_path.exists():
        return {"type": "FeatureCollection", "features": []}
    gdf = gpd.read_file(issues_path)
    return gdf.__geo_interface__


@router.post("/autofix")
async def apply_autofix(project_id: str):
    parcels_path = settings.data_dir / "processed" / "vectors" / "parcels_landuse.geojson"
    if not parcels_path.exists():
        parcels_path = settings.data_dir / "raw" / "synthetic_parcels.geojson"

    gdf = gpd.read_file(parcels_path)
    fixer = AutoFixEngine()
    healed_gdf, audit = fixer.auto_fix_all(gdf)

    out_path = settings.data_dir / "processed" / "validation" / "parcels_healed.geojson"
    healed_gdf.to_file(out_path, driver="GeoJSON")

    validator = CadastraValidator()
    post_report = validator.validate(healed_gdf)

    return {
        "status": "success",
        "repairs_count": len(audit),
        "audit": audit,
        "post_validation_summary": post_report["summary"],
    }
