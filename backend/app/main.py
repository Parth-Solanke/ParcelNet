from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routers import gt, jobs, layers, parcels, projects, validation
from backend.app.core.config import settings

app = FastAPI(
    title="CadastraAI - Urban Parcel Mapping & Cadastral Feature Extraction",
    description="AI-enabled urban cadastral mapping platform turning drone data into preliminary, GIS-ready parcel maps.",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root level and /api/v1 routers
api_v1 = APIRouter(prefix="/api/v1")
for r in [projects.router, jobs.router, layers.router, parcels.router, validation.router, gt.router]:
    app.include_router(r)
    api_v1.include_router(r)

app.include_router(api_v1)


@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/", response_class=HTMLResponse, tags=["dashboard"])
async def dashboard_index():
    index_file = settings.base_dir / "backend" / "app" / "static" / "dashboard" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>CadastraAI Backend Running. Open /docs for OpenAPI specifications.</h1>"


# Static data directory mount for serving rasters & COGs
settings.data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(settings.data_dir)), name="static")

dashboard_static = settings.base_dir / "backend" / "app" / "static" / "dashboard"
if dashboard_static.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_static), html=True), name="dashboard_static")
