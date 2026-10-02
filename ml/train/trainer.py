from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from backend.app.core.logging import logger
from ml.train.losses import MultiTaskCadastraLoss
from ml.train.metrics import MetricCalculator
from ml.train.model import MultiTaskCadastraNet


class ModelTrainer:
    """Trainer orchestrating multi-task model optimization, validation, and checkpointing."""

    def __init__(
        self,
        model: MultiTaskCadastraNet,
        device: Optional[torch.device] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        weights_dir: Optional[Path] = None,
    ):
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.criterion = MultiTaskCadastraLoss().to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.weights_dir = weights_dir or Path("weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        loss_components: Dict[str, float] = {
            "loss_buildings": 0.0,
            "loss_roads": 0.0,
            "loss_boundaries": 0.0,
            "loss_distance": 0.0,
            "loss_landuse": 0.0,
        }
        num_batches = len(dataloader)
        if num_batches == 0:
            return {"loss": 0.0}

        for images, targets in dataloader:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss, breakdown = self.criterion(preds, targets)
            loss.backward()

            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += breakdown["total_loss"]
            for k in loss_components:
                loss_components[k] += breakdown.get(k, 0.0)

        epoch_metrics = {k: v / num_batches for k, v in loss_components.items()}
        epoch_metrics["loss"] = total_loss / num_batches
        return epoch_metrics

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        bldg_ious = []
        road_ious = []
        boundary_f1s = []
        num_batches = len(dataloader)
        if num_batches == 0:
            return {"val_loss": 0.0, "val_boundary_f1": 0.0}

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                preds = self.model(images)
                loss, breakdown = self.criterion(preds, targets)
                total_loss += breakdown["total_loss"]

                # Convert to numpy for metric calculation
                b_pred = (torch.sigmoid(preds["buildings"]) > 0.5).cpu().numpy()
                b_gt = (targets["buildings"] > 0.5).cpu().numpy()
                r_pred = (torch.sigmoid(preds["roads"]) > 0.5).cpu().numpy()
                r_gt = (targets["roads"] > 0.5).cpu().numpy()
                bound_pred = (torch.sigmoid(preds["boundaries"]) > 0.5).cpu().numpy()
                bound_gt = (targets["boundaries"] > 0.5).cpu().numpy()

                for i in range(images.shape[0]):
                    bldg_ious.append(MetricCalculator.compute_binary_metrics(b_pred[i, 0], b_gt[i, 0])["iou"])
                    road_ious.append(MetricCalculator.compute_binary_metrics(r_pred[i, 0], r_gt[i, 0])["iou"])
                    bound_metrics = MetricCalculator.compute_boundary_f1_with_tolerance(
                        bound_pred[i, 0], bound_gt[i, 0], tolerance_px=3
                    )
                    boundary_f1s.append(bound_metrics["boundary_f1"])

        return {
            "val_loss": total_loss / num_batches,
            "val_building_iou": float(sum(bldg_ious) / max(len(bldg_ious), 1)),
            "val_road_iou": float(sum(road_ious) / max(len(road_ious), 1)),
            "val_boundary_f1": float(sum(boundary_f1s) / max(len(boundary_f1s), 1)),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        checkpoint_name: str = "cadastra_model_best.pt",
    ) -> Dict[str, Any]:
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        best_val_f1 = -1.0
        best_metrics: Dict[str, Any] = {}
        history = []

        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            log_str = (
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Bldg IoU: {val_metrics['val_building_iou']:.4f} | "
                f"Road IoU: {val_metrics['val_road_iou']:.4f} | "
                f"Bound F1: {val_metrics['val_boundary_f1']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            logger.info(log_str)

            record = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr": current_lr,
            }
            history.append(record)

            # Checkpoint best model on validation boundary F1
            if val_metrics["val_boundary_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_boundary_f1"]
                best_metrics = record

                ckpt_path = self.weights_dir / checkpoint_name
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": val_metrics,
                    },
                    ckpt_path,
                )
                metrics_path = self.weights_dir / f"{Path(checkpoint_name).stem}_metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=2)
                logger.info(f"-> Saved new best checkpoint to {ckpt_path} (Bound F1: {best_val_f1:.4f})")

        return {"best_metrics": best_metrics, "history": history}
