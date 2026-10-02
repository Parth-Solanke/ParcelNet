from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import uuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.dtos import ProjectCreateDTO, ProjectResponseDTO
from backend.app.services.pipeline_orchestrator import PipelineOrchestrator

router = APIRouter(prefix="/projects", tags=["projects"])

# In-memory store for projects (syncable with DB)
PROJECTS_STORE: Dict[str, Dict[str, Any]] = {}
JOBS_STORE: Dict[str, Dict[str, Any]] = {}


class PipelineRunRequest(BaseModel):
    ori_filename: str = "synthetic_ori.tif"
    dsm_filename: Optional[str] = "synthetic_dsm.tif"
    dtm_filename: Optional[str] = "synthetic_dtm.tif"
    vector_filename: Optional[str] = "synthetic_parcels.geojson"
    auto_heal: bool = True


@router.post("", response_model=Dict[str, Any])
async def create_project(payload: ProjectCreateDTO):
    project_id = str(uuid.uuid4())
    proj_dir = settings.data_dir / "projects" / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    project_data = {
        "id": project_id,
        "name": payload.name,
        "crs_epsg": payload.crs_epsg,
        "status": "created",
        "created_at": "2026-10-02T19:00:00",
        "work_dir": str(proj_dir),
    }
    PROJECTS_STORE[project_id] = project_data
    return project_data


@router.get("", response_model=List[Dict[str, Any]])
async def list_projects():
    return list(PROJECTS_STORE.values())


@router.get("/{project_id}", response_model=Dict[str, Any])
async def get_project(project_id: str):
    if project_id not in PROJECTS_STORE:
        raise HTTPException(status_code=404, detail="Project not found")
    return PROJECTS_STORE[project_id]


@router.post("/{project_id}/upload")
async def upload_project_file(
    project_id: str,
    file: UploadFile = File(...),
    file_type: str = Form("ORI"),  # ORI, DSM, DTM, VECTOR
):
    if project_id not in PROJECTS_STORE:
        raise HTTPException(status_code=404, detail="Project not found")

    proj_dir = Path(PROJECTS_STORE[project_id]["work_dir"]) / "uploads"
    proj_dir.mkdir(parents=True, exist_ok=True)

    dest_path = proj_dir / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "project_id": project_id,
        "filename": file.filename,
        "file_type": file_type,
        "saved_path": str(dest_path),
    }


def execute_pipeline_background(job_id: str, project_id: str, req: PipelineRunRequest):
    JOBS_STORE[job_id]["status"] = "running"
    proj_dir = Path(PROJECTS_STORE[project_id]["work_dir"])
    uploads_dir = proj_dir / "uploads"

    def on_progress(p: float, msg: str):
        JOBS_STORE[job_id]["progress"] = round(p * 100.0, 1)
        JOBS_STORE[job_id]["log"].append(msg)

    orchestrator = PipelineOrchestrator(
        project_id=project_id, work_dir=proj_dir, progress_callback=on_progress
    )

    # Resolve paths (either from uploads or data/raw demo data)
    ori = uploads_dir / req.ori_filename
    if not ori.exists():
        ori = settings.data_dir / "raw" / req.ori_filename

    dsm = uploads_dir / (req.dsm_filename or "")
    if not dsm.exists() and req.dsm_filename:
        dsm = settings.data_dir / "raw" / req.dsm_filename

    dtm = uploads_dir / (req.dtm_filename or "")
    if not dtm.exists() and req.dtm_filename:
        dtm = settings.data_dir / "raw" / req.dtm_filename

    vec = uploads_dir / (req.vector_filename or "")
    if not vec.exists() and req.vector_filename:
        vec = settings.data_dir / "raw" / req.vector_filename

    weights = settings.weights_dir / "cadastra_best.pt"

    try:
        results = orchestrator.run_pipeline(
            ori_path=ori,
            dsm_path=dsm if dsm and dsm.exists() else None,
            dtm_path=dtm if dtm and dtm.exists() else None,
            existing_parcels_path=vec if vec and vec.exists() else None,
            model_weights_path=weights if weights.exists() else None,
            auto_heal=req.auto_heal,
        )
        JOBS_STORE[job_id]["status"] = "completed"
        JOBS_STORE[job_id]["progress"] = 100.0
        JOBS_STORE[job_id]["results"] = results
        PROJECTS_STORE[project_id]["status"] = "ready"
    except Exception as e:
        logger.error(f"Pipeline failed for project {project_id}: {e}", exc_info=True)
        JOBS_STORE[job_id]["status"] = "failed"
        JOBS_STORE[job_id]["log"].append(f"ERROR: {str(e)}")


@router.post("/{project_id}/run")
async def run_project_pipeline(
    project_id: str,
    req: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    if project_id not in PROJECTS_STORE:
        raise HTTPException(status_code=404, detail="Project not found")

    job_id = str(uuid.uuid4())
    JOBS_STORE[job_id] = {
        "id": job_id,
        "project_id": project_id,
        "type": "full_pipeline",
        "status": "pending",
        "progress": 0.0,
        "log": ["Job registered"],
        "results": {},
    }

    background_tasks.add_task(execute_pipeline_background, job_id, project_id, req)
    return {"job_id": job_id, "status": "pending", "project_id": project_id}
