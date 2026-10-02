import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_create_and_list_projects(tmp_path):
    # 1. Create project
    payload = {
        "name": "Test Urban Ward 7",
        "description": "Validation test project",
        "crs_epsg": 32643,
    }
    create_res = client.post("/projects", json=payload)
    assert create_res.status_code == 200
    proj_data = create_res.json()
    assert "id" in proj_data
    assert proj_data["name"] == "Test Urban Ward 7"
    project_id = proj_data["id"]

    # 2. List projects
    list_res = client.get("/projects")
    assert list_res.status_code == 200
    projects = list_res.json()
    assert any(p["id"] == project_id for p in projects)

    # 3. Get single project
    get_res = client.get(f"/projects/{project_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Test Urban Ward 7"

    # Also test api/v1 prefix
    v1_res = client.get("/api/v1/projects")
    assert v1_res.status_code == 200


def test_layers_query_and_bbox():
    # Fetch layers for default project
    res = client.get("/projects/demo_project/layers/parcels")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data

    # Test BBOX filtering
    res_bbox = client.get(
        "/projects/demo_project/layers/parcels?minx=500000&miny=3000000&maxx=500050&maxy=3000050"
    )
    assert res_bbox.status_code == 200
    bbox_data = res_bbox.json()
    assert bbox_data["type"] == "FeatureCollection"


def test_parcel_patch():
    # Update parcel attributes with optimistic concurrency version
    patch_payload = {
        "version": 1,
        "landuse": "Commercial",
        "status": "edited",
    }
    res = client.patch("/parcels/PARCEL_0001", json=patch_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["parcel_id"] == "PARCEL_0001"
    assert data["updated"]["landuse"] == "Commercial"
    assert data["updated"]["version"] == 2


def test_validation_autofix_trigger():
    res = client.post("/projects/demo_project/validation/autofix")
    assert res.status_code == 200
    report = res.json()
    assert report["status"] == "success"
    assert "repairs_count" in report
    assert "post_validation_summary" in report


def test_layers_export_download():
    res = client.get("/projects/demo_project/layers/export/download?format=geojson&layer=parcels")
    assert res.status_code == 200
    assert "application/geo+json" in res.headers.get("content-type", "")
