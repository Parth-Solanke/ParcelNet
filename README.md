# CadastraAI 🛰️📐
**AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System**

CadastraAI is a high-performance GeoAI platform that transforms high-resolution drone imagery (2–5 cm GSD), Digital Surface Models (DSM), and Digital Terrain Models (DTM) into GIS-ready cadastral parcel maps, building footprints, road networks, and land-use classifications with automated topological validation.

---

## 🏗️ Architecture & Pipeline Flow

```
[Drone ORI (RGB)] + [DSM] + [DTM] + [Existing GIS / GT Points]
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ MODULE 1: Ingestion & Georeferencing QC                │
│  - Cloud-Optimized GeoTIFF (COG) conversion            │
│  - Metric CRS Reprojection & Resampling to Master Grid │
│  - Bounding Overlap & Alignment Verification           │
│  - Vector make_valid Repair & Affine Residual Checks   │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ MODULE 2: Pre-processing & Feature Stacking            │
│  - nDSM Computation: max(DSM - DTM, 0)                │
│  - Noise Filtering (<0.5m) & Median Spike Smoothing    │
│  - Auxiliary Channels: Slope, Hillshade, Excess Green  │
│  - Radiometric Normalization (2-98% Stretch & CLAHE)   │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ MODULE 3: Tiling Engine & Overlap Blending             │
│  - Sliding Window Tiling (1024px, 25% overlap)         │
│  - Tile Index GeoJSON / Metadata Generation            │
│  - 2D Hann / Cosine Window Seam Blending Map           │
│  - O(1) Memory Windowed Stitching & Reconstruction     │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Phase 1)

### 1. Environment Setup
```bash
# Clone the repository and enter directory
cd cadastra-ai

# Install dependencies in editable mode
pip install -e .
```

### 2. Generate Deterministic Sample Drone Dataset
```bash
python scripts/generate_sample_data.py
```
Outputs high-resolution synthetic drone data in `data/raw/`:
- `data/raw/synthetic_ori.tif` (RGB GeoTIFF, 5cm GSD)
- `data/raw/synthetic_dsm.tif` (DSM elevation model with buildings and roads)
- `data/raw/synthetic_dtm.tif` (DTM ground terrain)
- `data/raw/synthetic_parcels.geojson` (Cadastral ground truth polygons)

### 3. Module 1: Ingest & Georeference QC
```bash
python scripts/ingest.py \
  --ori data/raw/synthetic_ori.tif \
  --dsm data/raw/synthetic_dsm.tif \
  --dtm data/raw/synthetic_dtm.tif \
  --vector data/raw/synthetic_parcels.geojson \
  --output-dir data/processed
```
Outputs Cloud-Optimized GeoTIFFs (`ori_cog.tif`, `dsm_cog.tif`, `dtm_cog.tif`), repaired vector parcels, and `ingestion_qc_report.json`.

### 4. Module 2: Pre-process & Build Feature Stack
```bash
python scripts/preprocess.py \
  --ori data/processed/ori_cog.tif \
  --dsm data/processed/dsm_cog.tif \
  --dtm data/processed/dtm_cog.tif \
  --channels rgb+ndsm \
  --output data/processed/stack_4ch.tif
```
Computes normalized $nDSM$, removes ground noise, and packages a model-ready 4-channel input stack $[R, G, B, \text{nDSM}]$.

### 5. Module 3: Sliding-Window Tiling & Seam Blending
```bash
# Tile raster into chips & generate spatial index
python scripts/tile.py tile \
  --raster data/processed/stack_4ch.tif \
  --tile-size 256 \
  --overlap 0.25 \
  --out-index data/tiles/tile_index.geojson \
  --out-dir data/tiles/chips

# Stitch back seamlessly with Hann window blending
python scripts/tile.py stitch \
  --raster data/processed/stack_4ch.tif \
  --tile-size 256 \
  --overlap 0.25 \
  --output data/processed/reconstructed_stack.tif
```

### 5. Module 4: Build Training Dataset & Rasterize Labels
```bash
python scripts/build_dataset.py \
  --stack data/processed/stack_4ch.tif \
  --parcels data/raw/synthetic_parcels.geojson \
  --chip-size 256 \
  --stride 128 \
  --output-dir data/tiles/dataset
```
Extracts chips, dilates boundaries (2-3px), computes Euclidean distance transform maps, and generates a spatial-block-split `dataset_manifest.json`.

### 6. Module 5: Train Multi-Task Model & Tiled Inference
```bash
# Train multi-task CadastraNet with AdamW and Cosine Annealing LR
python scripts/train.py \
  --dataset-dir data/tiles/dataset \
  --epochs 10 \
  --batch-size 4 \
  --lr 0.0003 \
  --checkpoint-name cadastra_best.pt

# Run sliding-window inference with Test-Time Augmentation (TTA) and Hann blending
python scripts/infer.py \
  --stack data/processed/stack_4ch.tif \
  --weights weights/cadastra_best.pt \
  --tile-size 256 \
  --overlap 0.25 \
  --output-dir data/processed/predictions
```
### 7. Module 6: Vectorize Features & Orthogonalize Buildings
```bash
python scripts/vectorize.py \
  --buildings-prob data/processed/predictions/buildings_prob.tif \
  --roads-prob data/processed/predictions/roads_prob.tif \
  --ndsm data/processed/dsm_cog.tif \
  --output-dir data/processed/vectors
```
Produces `buildings.geojson` (dominant-angle snapped, median nDSM heights attached), `roads_centerlines.geojson` (skeletonized, spurs pruned), and `roads_polygons.geojson`.

### 8. Module 8: Generate Topology & Clean Parcels
```bash
python scripts/topology.py \
  --boundaries-prob data/processed/predictions/boundaries_prob.tif \
  --roads data/processed/vectors/roads_polygons.geojson \
  --buildings data/processed/vectors/buildings.geojson \
  --output data/processed/vectors/parcels.geojson
```
Builds noded line network, polygonizes closed faces, filters road faces, merges thin slivers into longest-border neighbours, and computes compactness and confidence scores.

### 9. Module 7: Object-Based Land-Use Refinement
```bash
python scripts/landuse.py \
  --parcels data/processed/vectors/parcels.geojson \
  --landuse-raster data/processed/predictions/landuse_prob.tif \
  --ndsm data/processed/dsm_cog.tif \
  --output data/processed/vectors/parcels_landuse.geojson
```
Assigns majority land-use class per parcel with nDSM physical elevation consistency checks.

### 10. Module 9: Topology Validation & Automated Healing
```bash
python scripts/validate.py \
  --parcels data/processed/vectors/parcels_landuse.geojson \
  --fix \
  --output-dir data/processed/validation
```
Evaluates invalid geometries, overlaps, duplicates, and shape anomalies. Applies `AutoFixEngine` to clip overlaps and repair bowties into `parcels_healed.geojson`.

### 11. Module 10: Ground Truthing & Field Verification
```bash
python scripts/gt.py \
  --parcels data/processed/validation/parcels_healed.geojson \
  --gt-points data/raw/control_points.csv \
  --output-dir data/processed/gt
```
Computes boundary offset residuals, updates statuses (`auto` -> `verified`), and generates mobile GIS worklists (`field_verification_list.geojson` for QField / Mergin Maps).

---

## 🧪 Testing & Verification

Run the complete test suite with `pytest`:
```bash
pytest -v
```
All 35 unit tests verified across Modules 1 to 10:
- `tests/test_module1_ingestion.py`: CRS verification, invalid CRS rejection, bounding box IoU calculation, vector topology repair (`make_valid`).
- `tests/test_module2_preprocessing.py`: nDSM computation, noise thresholding, hillshade, slope, Excess Green, 2-98% radiometric stretch.
- `tests/test_module3_tiling.py`: Sliding-window coverage verification, 2D Hann window weighting, and tile-and-stitch identity test.
- `tests/test_module4_dataset.py`: Multi-task label rasterization, boundary distance transform, and spatial block splitting.
- `tests/test_module5_model.py`: Multi-task model forward pass, combined loss components, relaxed boundary-F1 metric, overfit smoke test, and tiled inference.
- `tests/test_module6_vectorization.py`: Building footprint orthogonalization, median height extraction, road skeletonization, and network graph.
- `tests/test_module7_landuse.py`: Object-based majority refinement and physical elevation sanity checks.
- `tests/test_module8_topology.py`: Noded line networks, closed face polygonization, and sliver merging.
- `tests/test_module9_validation.py`: Geometry validity, overlap detection, duplicate detection, shape anomalies, and AutoFixEngine healing.
- `tests/test_module10_gt.py`: Ground truthing CSV import, boundary offset matching, and mobile field verification export.

---

## 📂 Repository Layout

```
cadastra-ai/
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml         # Project CRS, tolerances, and model parameters
├── data/                    # (Gitignored) Raw, processed, and tile data
├── weights/                 # (Gitignored) Pretrained and fine-tuned checkpoints
├── backend/
│   └── app/
│       ├── core/            # Config (YAML + Pydantic), logging, database sessions
│       ├── models/          # PostGIS schema (SQLAlchemy 2.0) and Pydantic DTOs
│       └── services/
│           ├── ingestion/   # Module 1: Validators, Reprojector, COG converter, QC
│           ├── preprocessing/# Module 2: nDSM, Hillshade, Slope, Stacker
│           └── tiling/      # Module 3: Sliding window, 2D Hann blender, Stitcher
├── scripts/
│   ├── generate_sample_data.py # Sample drone dataset generator
│   ├── ingest.py            # Module 1 CLI
│   ├── preprocess.py        # Module 2 CLI
│   └── tile.py              # Module 3 CLI
└── tests/
    ├── fixtures/
    │   └── synthetic_drone_data.py
    ├── test_module1_ingestion.py
    ├── test_module2_preprocessing.py
    └── test_module3_tiling.py
```
