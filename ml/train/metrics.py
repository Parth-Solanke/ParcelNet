from typing import Dict, Tuple
import numpy as np
import torch
from scipy.ndimage import binary_dilation


class MetricCalculator:
    """Calculates GeoAI segmentation metrics: IoU, F1, Precision, Recall, and Relaxed Boundary-F1."""

    @staticmethod
    def compute_binary_metrics(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        eps: float = 1e-6,
    ) -> Dict[str, float]:
        """Calculates Precision, Recall, F1, and IoU for binary segmentation."""
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()

        precision = (intersection + eps) / (pred_sum + eps)
        recall = (intersection + eps) / (gt_sum + eps)
        f1 = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
        iou = (intersection + eps) / (union + eps)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
        }

    @staticmethod
    def compute_boundary_f1_with_tolerance(
        pred_boundary: np.ndarray,
        gt_boundary: np.ndarray,
        tolerance_px: int = 3,
        eps: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Calculates boundary precision, recall, and F1 within a spatial buffer tolerance.
        A predicted boundary pixel is correct if within `tolerance_px` of true boundary,
        and a true boundary pixel is detected if within `tolerance_px` of predicted boundary.
        """
        pred = pred_boundary.astype(bool)
        gt = gt_boundary.astype(bool)

        if not np.any(gt) and not np.any(pred):
            return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f1": 1.0}
        if not np.any(gt) or not np.any(pred):
            return {"boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0}

        struct = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), dtype=bool)

        # Buffer GT boundary
        gt_buffered = binary_dilation(gt, structure=struct)
        # Precision: fraction of predicted pixels within GT buffer
        matched_pred = np.logical_and(pred, gt_buffered).sum()
        precision = (matched_pred + eps) / (pred.sum() + eps)

        # Buffer Predicted boundary
        pred_buffered = binary_dilation(pred, structure=struct)
        # Recall: fraction of GT pixels within Predicted buffer
        matched_gt = np.logical_and(gt, pred_buffered).sum()
        recall = (matched_gt + eps) / (gt.sum() + eps)

        f1 = (2.0 * precision * recall) / (precision + recall + eps)

        return {
            "boundary_precision": float(precision),
            "boundary_recall": float(recall),
            "boundary_f1": float(f1),
        }
