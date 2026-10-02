#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from backend.app.core.logging import logger
from ml.datasets.cadastra_dataset import CadastraDataset
from ml.train.model import MultiTaskCadastraNet
from ml.train.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 5: Multi-Task Model Training CLI"
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/tiles/dataset"), help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"), help="Directory to save model weights")
    parser.add_argument("--checkpoint-name", type=str, default="cadastra_best.pt", help="Checkpoint filename")

    args = parser.parse_args()

    logger.info("Initializing CadastraAI Multi-Task Model Trainer...")
    train_dataset = CadastraDataset(args.dataset_dir, split="train", augment=True)
    val_dataset = CadastraDataset(args.dataset_dir, split="val", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiTaskCadastraNet(in_channels=4, pretrained=False)
    trainer = ModelTrainer(
        model=model,
        learning_rate=args.lr,
        weights_dir=args.weights_dir,
    )

    try:
        results = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            checkpoint_name=args.checkpoint_name,
        )
        logger.info(f"Training completed successfully! Best metrics: {results['best_metrics']}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
