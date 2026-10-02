from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss designed to address severe class imbalance in thin linear features
    like parcel boundary lines by weighting false negatives higher than false positives.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # weight on false positives
        self.beta = beta    # weight on false negatives (higher penalty for missing boundary)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        true_pos = (probs_flat * targets_flat).sum()
        false_neg = (targets_flat * (1.0 - probs_flat)).sum()
        false_pos = ((1.0 - targets_flat) * probs_flat).sum()

        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        return 1.0 - tversky


class MultiTaskCadastraLoss(nn.Module):
    """
    Unified multi-task loss function combining:
      - BCE + Dice for buildings
      - BCE + Dice for roads
      - Weighted BCE + Tversky + L1 Distance Loss for boundaries
      - CrossEntropy for landuse
    """

    def __init__(
        self,
        weight_buildings: float = 1.0,
        weight_roads: float = 1.0,
        weight_boundaries: float = 2.5,
        weight_distance: float = 0.5,
        weight_landuse: float = 1.0,
        boundary_pos_weight: float = 5.0,
    ):
        super().__init__()
        self.w_bldg = weight_buildings
        self.w_road = weight_roads
        self.w_bound = weight_boundaries
        self.w_dist = weight_distance
        self.w_lu = weight_landuse

        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.bce = nn.BCEWithLogitsLoss()
        self.bound_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([boundary_pos_weight]))
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Buildings loss
        bldg_bce = self.bce(preds["buildings"], targets["buildings"])
        bldg_dice = self.dice(preds["buildings"], targets["buildings"])
        loss_bldg = bldg_bce + bldg_dice

        # 2. Roads loss
        road_bce = self.bce(preds["roads"], targets["roads"])
        road_dice = self.dice(preds["roads"], targets["roads"])
        loss_road = road_bce + road_dice

        # 3. Boundaries loss: weighted BCE + Tversky + L1 on distance map
        device = preds["boundaries"].device
        if self.bound_bce.pos_weight.device != device:
            self.bound_bce.pos_weight = self.bound_bce.pos_weight.to(device)

        bound_bce = self.bound_bce(preds["boundaries"], targets["boundaries"])
        bound_tversky = self.tversky(preds["boundaries"], targets["boundaries"])
        loss_dist = self.l1(preds["distance_map"], targets["distance_map"])
        loss_bound = bound_bce + bound_tversky + (self.w_dist * loss_dist)

        # 4. Landuse classification loss
        loss_lu = self.ce(preds["landuse"], targets["landuse"])

        # Total combined loss
        total_loss = (
            self.w_bldg * loss_bldg
            + self.w_road * loss_road
            + self.w_bound * loss_bound
            + self.w_lu * loss_lu
        )

        loss_breakdown = {
            "total_loss": float(total_loss.item()),
            "loss_buildings": float(loss_bldg.item()),
            "loss_roads": float(loss_road.item()),
            "loss_boundaries": float(loss_bound.item()),
            "loss_distance": float(loss_dist.item()),
            "loss_landuse": float(loss_lu.item()),
        }

        return total_loss, loss_breakdown
