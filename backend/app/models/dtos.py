from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ProjectCreateDTO(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    crs_epsg: int = Field(32643, description="Projected EPSG code (e.g. UTM)")


class ProjectResponseDTO(BaseModel):
    id: str
    name: str
    crs_epsg: int
    bbox: Optional[List[float]] = None
    created_at: datetime
    status: str

    model_config = ConfigDict(from_attributes=True)


class RasterResponseDTO(BaseModel):
    id: str
    project_id: str
    type: Literal["ORI", "DSM", "DTM", "NDSM", "STACK"]
    path: str
    gsd: float
    bounds: List[float]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class JobResponseDTO(BaseModel):
    id: str
    project_id: str
    type: str
    status: str
    progress: float
    params_json: Dict[str, Any]
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    log: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ValidationIssueDTO(BaseModel):
    id: str
    project_id: str
    type: str
    severity: Literal["error", "warning", "info"]
    feature_ids: List[str]
    resolved: bool
    details: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ControlPointDTO(BaseModel):
    id: Optional[str] = None
    x: float
    y: float
    height: float
    accuracy_cm: float


class QCReportDTO(BaseModel):
    is_valid: bool
    project_crs: int
    raster_summaries: Dict[str, Dict[str, Any]]
    bounds_overlap_ratio: float
    alignment_verified: bool
    control_points_rmse_m: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
