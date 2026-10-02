from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from backend.app.api.routers.projects import JOBS_STORE

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str):
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS_STORE[job_id]
