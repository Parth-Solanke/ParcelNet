from pathlib import Path
import pytest

from backend.app.services.pipeline_orchestrator import PipelineOrchestrator
from tests.fixtures.synthetic_drone_data import generate_synthetic_drone_dataset


def test_end_to_end_pipeline(tmp_path: Path):
    """
    Integration Acceptance Test:
    Executes the entire CadastraAI GeoAI pipeline end-to-end:
    Ingest -> Preprocess -> Infer -> Vectorize -> Landuse -> Topology -> Validate -> Auto-fix.
    """
    raw_dir = tmp_path / "raw"
    synthetic_data = generate_synthetic_drone_dataset(raw_dir, width=150, height=150, gsd=0.05)

    proj_dir = tmp_path / "project_e2e"
    orchestrator = PipelineOrchestrator(project_id="test_e2e_001", work_dir=proj_dir)

    results = orchestrator.run_pipeline(
        ori_path=synthetic_data["ori"],
        dsm_path=synthetic_data["dsm"],
        dtm_path=synthetic_data["dtm"],
        existing_parcels_path=synthetic_data["vector"],
        auto_heal=True,
    )

    # 1. Ingestion verified
    assert "ingestion_qc" in results
    assert results["ingestion_qc"]["is_valid"] is True

    # 2. Preprocessing feature stack verified
    assert Path(results["stack_path"]).exists()

    # 3. Model predictions verified
    assert "buildings" in results["predictions"]
    assert "roads" in results["predictions"]
    assert "boundaries" in results["predictions"]
    assert "landuse" in results["predictions"]
    assert Path(results["predictions"]["boundaries"]).exists()

    # 4. Topology and Parcels verified
    assert Path(results["final_parcels_path"]).exists()
    assert results["parcels_count"] > 0

    # 5. Validation summary verified
    assert "validation" in results
    assert results["validation"]["clean_percentage"] >= 0.0
