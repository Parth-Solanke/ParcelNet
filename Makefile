.PHONY: setup run test demo evaluate clean

PYTHON = python
PYTEST = pytest
UVICORN = uvicorn

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install pytest sqlalchemy geoalchemy2 alembic torch torchvision albumentations

run:
	$(UVICORN) backend.app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTEST) -v

demo:
	@echo "--- 1. Generating Sample Drone Data ---"
	$(PYTHON) scripts/generate_sample_data.py
	@echo "--- 2. Module 1: Ingestion & QC ---"
	$(PYTHON) scripts/ingest.py --ori data/raw/synthetic_ori.tif --dsm data/raw/synthetic_dsm.tif --dtm data/raw/synthetic_dtm.tif --vector data/raw/synthetic_parcels.geojson --output-dir data/processed
	@echo "--- 3. Module 2: Feature Stacking ---"
	$(PYTHON) scripts/preprocess.py --ori data/processed/ori_cog.tif --dsm data/processed/dsm_cog.tif --dtm data/processed/dtm_cog.tif --channels rgb+ndsm --output data/processed/stack_4ch.tif
	@echo "--- 4. Module 3: Sliding Window Tiling ---"
	$(PYTHON) scripts/tile.py tile --raster data/processed/stack_4ch.tif --tile-size 256 --overlap 0.25 --out-index data/tiles/tile_index.geojson
	@echo "--- 5. Module 4: Dataset Build ---"
	$(PYTHON) scripts/build_dataset.py --stack data/processed/stack_4ch.tif --parcels data/raw/synthetic_parcels.geojson --chip-size 256 --stride 128 --output-dir data/tiles/dataset
	@echo "--- 6. Module 5: Train & Tiled Inference ---"
	$(PYTHON) scripts/train.py --dataset-dir data/tiles/dataset --epochs 2 --batch-size 2 --checkpoint-name cadastra_best.pt
	$(PYTHON) scripts/infer.py --stack data/processed/stack_4ch.tif --weights weights/cadastra_best.pt --tile-size 256 --overlap 0.25 --output-dir data/processed/predictions
	@echo "--- 7. Module 6: Vectorization ---"
	$(PYTHON) scripts/vectorize.py --buildings-prob data/processed/predictions/buildings_prob.tif --roads-prob data/processed/predictions/roads_prob.tif --ndsm data/processed/dsm_cog.tif --output-dir data/processed/vectors
	@echo "--- 8. Module 8: Topology Engine ---"
	$(PYTHON) scripts/topology.py --boundaries-prob data/processed/predictions/boundaries_prob.tif --roads data/processed/vectors/roads_polygons.geojson --buildings data/processed/vectors/buildings.geojson --output data/processed/vectors/parcels.geojson
	@echo "--- 9. Module 7: Land-Use Refinement ---"
	$(PYTHON) scripts/landuse.py --parcels data/raw/synthetic_parcels.geojson --landuse-raster data/processed/predictions/landuse_prob.tif --ndsm data/processed/dsm_cog.tif --output data/processed/vectors/parcels_landuse.geojson
	@echo "--- 10. Module 9: Validation & Auto-Healing ---"
	$(PYTHON) scripts/validate.py --parcels data/processed/vectors/parcels_landuse.geojson --fix --output-dir data/processed/validation
	@echo "--- 11. Module 10: Ground Truthing & Verification Worklist ---"
	$(PYTHON) scripts/gt.py --parcels data/processed/validation/parcels_healed.geojson --output-dir data/processed/gt
	@echo "--- 12. Module 13: Evaluation & Benchmarking ---"
	$(PYTHON) scripts/evaluate.py --pred data/processed/validation/parcels_healed.geojson --ref data/raw/synthetic_parcels.geojson --output-dir data/reports
	@echo "=== DEMO EXECUTION COMPLETE! Web Dashboard available at http://localhost:8000 ==="

evaluate:
	$(PYTHON) scripts/evaluate.py --pred data/processed/validation/parcels_healed.geojson --ref data/raw/synthetic_parcels.geojson --output-dir data/reports
