#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from backend.app.core.logging import logger
from ml.infer.tiled_infer import TiledPredictor
from ml.train.model import MultiTaskCadastraNet


def main():
    parser = argparse.ArgumentParser(
        description="CadastraAI Module 5: Multi-Task Tiled Inference CLI"
    )
    parser.add_argument("--stack", type=Path, required=True, help="Path to 4-channel input stack GeoTIFF")
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for inference (default: 512)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio (default: 0.25)")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/predictions"),
        help="Directory to save output probability GeoTIFFs",
    )

    args = parser.parse_args()

    logger.info("Initializing CadastraAI Tiled Predictor...")
    model = MultiTaskCadastraNet(in_channels=4, pretrained=False)

    if args.weights and args.weights.exists():
        logger.info(f"Loading checkpoint weights from: {args.weights}")
        checkpoint = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.info("No weights provided; running inference with baseline initialized network.")

    predictor = TiledPredictor(
        model=model,
        tile_size_px=args.tile_size,
        overlap_ratio=args.overlap,
        use_tta=not args.no_tta,
    )

    try:
        output_paths = predictor.predict_raster(
            stack_path=args.stack,
            output_dir=args.output_dir,
        )
        logger.info("Inference completed successfully! Outputs:")
        for k, v in output_paths.items():
            logger.info(f"  {k}: {v}")
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
