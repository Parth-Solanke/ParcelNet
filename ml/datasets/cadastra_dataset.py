from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class CadastraDataset(Dataset):
    """
    Multi-task PyTorch Dataset for CadastraAI.
    Loads 4-channel feature chips [R, G, B, nDSM] and returns synchronized
    targets for buildings, roads, boundaries, distance maps, and landuse.
    """

    def __init__(
        self,
        dataset_dir: Path,
        split: str = "train",
        manifest_path: Optional[Path] = None,
        augment: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.augment = augment

        m_path = manifest_path or (self.dataset_dir / "dataset_manifest.json")
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found: {m_path}")

        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.samples = manifest["splits"].get(split, [])

        # Spatial augmentations applied across all channels and masks simultaneously
        self.spatial_transform = (
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
            if augment
            else None
        )

        # Radiometric augmentations applied ONLY to RGB channels (0, 1, 2)
        self.rgb_transform = (
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ])
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample_info = self.samples[idx]
        img_path = self.dataset_dir / sample_info["image_path"]
        mask_path = self.dataset_dir / sample_info["mask_path"]

        # Load 4-channel image (C, H, W)
        image = np.load(img_path)  # shape (4, H, W), float32 [0.0, 1.0]

        # Load multi-task masks
        masks_data = np.load(mask_path)
        bldg_mask = masks_data["buildings"].astype(np.float32)
        road_mask = masks_data["roads"].astype(np.float32)
        bound_mask = masks_data["boundaries"].astype(np.float32)
        dist_map = masks_data["distance_map"].astype(np.float32)
        landuse_mask = masks_data["landuse"].astype(np.int64)

        # Transpose image to (H, W, C) for albumentations
        img_hwc = np.transpose(image, (1, 2, 0))

        if self.augment:
            # 1. Apply RGB-only radiometric jitter (leave 4th channel nDSM intact)
            if self.rgb_transform is not None:
                rgb_part = (img_hwc[:, :, :3] * 255.0).astype(np.uint8)
                augmented_rgb = self.rgb_transform(image=rgb_part)["image"].astype(np.float32) / 255.0
                img_hwc[:, :, :3] = augmented_rgb

            # 2. Apply synchronized spatial transformations
            if self.spatial_transform is not None:
                # Combine masks for joint geometric transformation
                # dist_map is continuous, others discrete
                stacked_masks = np.stack(
                    [bldg_mask, road_mask, bound_mask, dist_map, landuse_mask.astype(np.float32)],
                    axis=-1,
                )
                transformed = self.spatial_transform(image=img_hwc, mask=stacked_masks)
                img_hwc = transformed["image"]
                stacked_masks = transformed["mask"]

                bldg_mask = (stacked_masks[:, :, 0] > 0.5).astype(np.float32)
                road_mask = (stacked_masks[:, :, 1] > 0.5).astype(np.float32)
                bound_mask = (stacked_masks[:, :, 2] > 0.5).astype(np.float32)
                dist_map = np.clip(stacked_masks[:, :, 3], 0.0, 1.0).astype(np.float32)
                landuse_mask = np.round(stacked_masks[:, :, 4]).astype(np.int64)

        # Transpose back to (C, H, W)
        image_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)

        # Convert to PyTorch Tensors
        x_tensor = torch.from_numpy(image_chw)
        targets = {
            "buildings": torch.from_numpy(bldg_mask).unsqueeze(0),       # (1, H, W)
            "roads": torch.from_numpy(road_mask).unsqueeze(0),               # (1, H, W)
            "boundaries": torch.from_numpy(bound_mask).unsqueeze(0),         # (1, H, W)
            "distance_map": torch.from_numpy(dist_map).unsqueeze(0),         # (1, H, W)
            "landuse": torch.from_numpy(landuse_mask),                       # (H, W)
        }

        return x_tensor, targets
