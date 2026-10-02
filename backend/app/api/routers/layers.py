from pathlib import Path
from typing import Any, Dict, List, Optional
import geopandas as gpd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from shapely.geometry import box

from backend.app.api.routers.projects import PROJECTS_STORE
from backend.app.core.config import settings

router = APIRouter(prefix="/projects/{project_id}/layers", tags=["layers"])


@router.get("/{layer_name}")
async def get_layer_geojson(
    project_id: str,
    layer_name: str,
    minx: Optional[float] = None,
    miny: Optional[float] = None,
    maxx: Optional[float] = None,
    maxy: Optional[float] = None,
):
    """Returns layer as GeoJSON FeatureCollection with optional bounding box filtering."""
    if project_id not in PROJECTS_STORE:
        # Fallback to shared processed vectors for testing
        work_dir = settings.data_dir / "processed" / "vectors"
    else:
        work_dir = Path(PROJECTS_STORE[project_id]["work_dir"]) / "vectors"

    file_mapping = {
        "parcels": "parcels_final.geojson",
        "parcels_raw": "parcels.geojson",
        "parcels_landuse": "parcels_landuse.geojson",
        "buildings": "buildings.geojson",
        "roads": "roads_centerlines.geojson",
        "road_polygons": "roads_polygons.geojson",
    }

    target_file = file_mapping.get(layer_name)
    if not target_file:
        raise HTTPException(status_code=400, detail=f"Unknown layer: {layer_name}")

    target_path = work_dir / target_file
    if not target_path.exists():
        # Fallback to general processed folder
        target_path = settings.data_dir / "processed" / "vectors" / target_file
        if not target_path.exists():
            target_path = settings.data_dir / "raw" / "synthetic_parcels.geojson"

    if not target_path.exists():
        return {"type": "FeatureCollection", "features": []}

    gdf = gpd.read_file(target_path)

    # Apply Bbox filter if provided
    if all(v is not None for v in [minx, miny, maxx, maxy]):
        bbox_geom = box(minx, miny, maxx, maxy)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)]

    return gdf.__geo_interface__


@router.get("/export/download")
async def export_layers_download(
    project_id: str,
    format: str = Query("geojson", regex="^(geojson|gpkg|shp)$"),
    layer: str = "parcels",
):
    """Exports and downloads processed GIS layer in requested format (GeoJSON, GeoPackage, Shapefile)."""
    geojson_data = await get_layer_geojson(project_id, layer)
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    gdf.set_crs(epsg=32643, inplace=True)

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
        out_file = export_dir / f"{layer}_{project_id[:8]}.zip"
        shp_dir = export_dir / f"{layer}_shp"
        shp_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(shp_dir / f"{layer}.shp")
        import shutil
        shutil.make_archive(str(export_dir / f"{layer}_{project_id[:8]}"), "zip", shp_dir)
        return FileResponse(out_file, media_type="application/zip", filename=out_file.name)
