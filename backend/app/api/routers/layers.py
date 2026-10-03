from pathlib import Path
from typing import Any, Dict, Optional
import math
import geopandas as gpd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from shapely.geometry import box

from backend.app.api.routers.projects import PROJECTS_STORE
from backend.app.core.config import settings

router = APIRouter(prefix="/projects/{project_id}/layers", tags=["layers"])

FILE_MAPPING: Dict[str, str] = {
    "parcels":         "parcels_final.geojson",
    "parcels_raw":     "parcels.geojson",
    "parcels_landuse": "parcels_landuse.geojson",
    "buildings":       "buildings.geojson",
    "roads":           "roads_centerlines.geojson",
    "road_polygons":   "roads_polygons.geojson",
}


def _clean_nan(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON safety."""
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


def _resolve_layer_path(project_id: str, layer_name: str) -> Optional[Path]:
    """Return the best existing path for a given layer, or None."""
    if project_id not in PROJECTS_STORE:
        work_dir = settings.data_dir / "processed" / "vectors"
    else:
        work_dir = Path(PROJECTS_STORE[project_id]["work_dir"]) / "vectors"

    target_file = FILE_MAPPING.get(layer_name)
    if not target_file:
        return None

    # Priority order: project work_dir → processed/vectors → raw synthetic
    candidates = [
        work_dir / target_file,
        settings.data_dir / "processed" / "vectors" / target_file,
        settings.data_dir / "raw" / "synthetic_parcels.geojson",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_and_reproject(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame and ensure it is in EPSG:4326 for web rendering."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=32643)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


@router.get("/{layer_name}", response_model=None)
async def get_layer_geojson(
    project_id: str,
    layer_name: str,
    minx: Optional[float] = None,
    miny: Optional[float] = None,
    maxx: Optional[float] = None,
    maxy: Optional[float] = None,
):
    """Returns layer as a JSON-safe WGS84 GeoJSON FeatureCollection."""
    if layer_name not in FILE_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown layer '{layer_name}'. Valid: {list(FILE_MAPPING)}")

    target_path = _resolve_layer_path(project_id, layer_name)
    if target_path is None:
        return {"type": "FeatureCollection", "features": []}

    gdf = _load_and_reproject(target_path)

    # BBox filter (expects WGS84 coordinates after reprojection)
    if all(v is not None for v in [minx, miny, maxx, maxy]):
        bbox_geom = box(minx, miny, maxx, maxy)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)]

    if len(gdf) == 0:
        return _safe_json_response({"type": "FeatureCollection", "features": []})

    return _safe_json_response(gdf.__geo_interface__)


@router.get("/export/download")
async def export_layers_download(
    project_id: str,
    format: str = Query("geojson", pattern="^(geojson|gpkg|shp)$"),
    layer: str = "parcels",
):
    """Exports and downloads a processed GIS layer in GeoJSON, GeoPackage, or Shapefile format."""
    if layer not in FILE_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown layer '{layer}'. Valid: {list(FILE_MAPPING)}")

    target_path = _resolve_layer_path(project_id, layer)
    if target_path is None:
        raise HTTPException(status_code=404, detail=f"No data found for layer '{layer}' in project '{project_id}'")

    gdf = gpd.read_file(target_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=32643)

    export_dir = settings.data_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if format == "geojson":
        out_file = export_dir / f"{layer}_{project_id[:8]}.geojson"
        gdf.to_file(out_file, driver="GeoJSON")
        return FileResponse(out_file, media_type="application/geo+json", filename=out_file.name)

    elif format == "gpkg":
        out_file = export_dir / f"{layer}_{project_id[:8]}.gpkg"
        gdf.to_file(out_file, driver="GPKG")
        return FileResponse(out_file, media_type="application/geopackage+sqlite3", filename=out_file.name)

    elif format == "shp":
        import shutil
        shp_dir = export_dir / f"{layer}_shp_{project_id[:8]}"
        shp_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(shp_dir / f"{layer}.shp")
        zip_base = str(export_dir / f"{layer}_{project_id[:8]}")
        shutil.make_archive(zip_base, "zip", shp_dir)
        out_file = Path(zip_base + ".zip")
        return FileResponse(out_file, media_type="application/zip", filename=out_file.name)
