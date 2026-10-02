from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shapely.geometry import shape

from backend.app.services.validation.rules import InvalidGeometryRule

router = APIRouter(prefix="/parcels", tags=["parcels"])

# In-memory parcel edit store
PARCEL_EDITS: Dict[str, Dict[str, Any]] = {}


class ParcelPatchDTO(BaseModel):
    version: int
    geometry: Optional[Dict[str, Any]] = None
    status: Optional[str] = None  # auto, edited, verified, rejected
    landuse: Optional[str] = None
    attrs: Optional[Dict[str, Any]] = None


@router.patch("/{parcel_id}")
async def patch_parcel(parcel_id: str, patch: ParcelPatchDTO):
    """
    Applies geometry or attribute edits with optimistic concurrency locking.
    Validates updated geometry.
    """
    current = PARCEL_EDITS.get(parcel_id, {"version": 1, "id": parcel_id})

    # Optimistic locking check
    if patch.version != current["version"]:
        raise HTTPException(
            status_code=409,
            detail=f"Conflict: Expected parcel version {current['version']}, but got {patch.version}.",
        )

    # Validate geometry if updated
    if patch.geometry:
        geom = shape(patch.geometry)
        if not geom.is_valid:
            raise HTTPException(status_code=400, detail="Updated geometry is invalid or self-intersecting.")
        current["geometry"] = patch.geometry

    if patch.status:
        current["status"] = patch.status
    if patch.landuse:
        current["landuse"] = patch.landuse
    if patch.attrs:
        current["attrs"] = patch.attrs

    current["version"] += 1
    PARCEL_EDITS[parcel_id] = current

    return {
        "status": "success",
        "parcel_id": parcel_id,
        "updated": current,
    }
