from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from backend.app.core.logging import logger
from ml.datasets.label_rasterizer import LabelRasterizer


class DatasetBuilder:
    """Builds training chip dataset with multi-task targets, spatial block split, and manifest."""

    def __init__(
        self,
        chip_size_px: int = 512,
        stride_px: int = 384,
        max_nodata_ratio: float = 0.70,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.chip_size = chip_size_px
        self.stride = stride_px
        self.max_nodata_ratio = max_nodata_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rasterizer = LabelRasterizer()

    def determine_spatial_split(
        self,
        centroid_x: float,
        centroid_y: float,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> str:
        """Assigns chip to train/val/test based on geographic spatial blocks to avoid leakage."""
        rel_x = (centroid_x - min_x) / max(max_x - min_x, 1e-5)
        rel_y = (centroid_y - min_y) / max(max_y - min_y, 1e-5)

        # 4 quadrant blocks or geographic bands
        # Train: bottom 70% or West + South
        if rel_x < 0.70:
            return "train"
        elif rel_y < 0.50:
            return "val"
        else:
            return "test"

    def build_dataset(
        self,
        stack_path: Path,
        output_dir: Path,
        parcels_gdf: Optional[gpd.GeoDataFrame] = None,
        buildings_gdf: Optional[gpd.GeoDataFrame] = None,
        roads_gdf: Optional[gpd.GeoDataFrame] = None,
        landuse_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> Path:
        """Extracts chips, rasterizes multi-task labels, and writes dataset manifest."""
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Any] = {
            "chip_size": self.chip_size,
            "classes": ["buildings", "roads", "parcel_boundary", "parcel_interior", "landuse"],
            "splits": {"train": [], "val": [], "test": []},
            "class_counts": {"buildings": 0, "roads": 0, "boundaries": 0, "interiors": 0},
            "total_chips": 0,
        }

        with rasterio.open(stack_path) as src:
            width, height = src.width, src.height
            transform = src.transform
            full_minx, full_miny, full_maxx, full_maxy = src.bounds

            chip_idx = 0
            for r in range(0, height, self.stride):
                for c in range(0, width, self.stride):
                    win_h = min(self.chip_size, height - r)
                    win_w = min(self.chip_size, width - c)
                    if win_h < self.chip_size or win_w < self.chip_size:
                        continue  # Skip partial border chips for clean square training

                    win = Window(col_off=c, row_off=r, width=win_w, height=win_h)
                    chip_img = src.read(window=win)  # (C, H, W)
                    chip_transform = rasterio.windows.transform(win, transform)

                    # Check nodata ratio
                    nodata_ratio = float(np.mean(chip_img == 0.0))
                    if nodata_ratio > self.max_nodata_ratio:
                        continue

                    # Centroid coordinates for spatial block split
                    center_col = c + win_w / 2.0
                    center_row = r + win_h / 2.0
                    cent_x, cent_y = transform * (center_col, center_row)

                    split = self.determine_spatial_split(
                        cent_x, cent_y, full_minx, full_miny, full_maxx, full_maxy
                    )

                    # Rasterize multi-task targets
                    chip_shape = (win_h, win_w)
                    bldg_mask = self.rasterizer.rasterize_polygons(buildings_gdf, chip_shape, chip_transform)
                    road_mask = self.rasterizer.rasterize_polygons(roads_gdf, chip_shape, chip_transform)
                    interior_mask = self.rasterizer.rasterize_polygons(parcels_gdf, chip_shape, chip_transform)
                    boundary_mask, dist_map = self.rasterizer.rasterize_boundaries_and_distance(
                        parcels_gdf, chip_shape, chip_transform
                    )
                    landuse_mask = self.rasterizer.rasterize_landuse(landuse_gdf, chip_shape, chip_transform)

                    # Save image and targets
                    chip_name = f"chip_{chip_idx:05d}"
                    img_file = images_dir / f"{chip_name}.npy"
                    mask_file = masks_dir / f"{chip_name}.npz"

                    np.save(img_file, chip_img.astype(np.float32))
                    np.savez_compressed(
                        mask_file,
                        buildings=bldg_mask,
                        roads=road_mask,
                        boundaries=boundary_mask,
                        interiors=interior_mask,
                        distance_map=dist_map,
                        landuse=landuse_mask,
                    )

                    manifest["splits"][split].append({
                        "id": chip_name,
                        "image_path": str(img_file.relative_to(output_dir)),
                        "mask_path": str(mask_file.relative_to(output_dir)),
                    })

                    # Update stats
                    manifest["class_counts"]["buildings"] += int(np.sum(bldg_mask > 0))
                    manifest["class_counts"]["roads"] += int(np.sum(road_mask > 0))
                    manifest["class_counts"]["boundaries"] += int(np.sum(boundary_mask > 0))
                    manifest["class_counts"]["interiors"] += int(np.sum(interior_mask > 0))
                    manifest["total_chips"] += 1
                    chip_idx += 1

        # Ensure validation split is not empty on small demo sets
        if len(manifest["splits"]["val"]) == 0 and len(manifest["splits"]["train"]) > 1:
            manifest["splits"]["val"].append(manifest["splits"]["train"].pop())

        manifest_path = output_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Dataset built successfully: {manifest['total_chips']} chips "
            f"(train: {len(manifest['splits']['train'])}, val: {len(manifest['splits']['val'])}, "
            f"test: {len(manifest['splits']['test'])}) -> {manifest_path}"
        )
        return manifest_path
