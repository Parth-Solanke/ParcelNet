import numpy as np


class WindowBlender:
    """Generates 2D tapering weight windows (Hann or Cosine) for seamless tile stitching."""

    @staticmethod
    def get_hann_window_2d(height: int, width: int, min_weight: float = 1e-4) -> np.ndarray:
        """
        Generates a 2D Hann window of shape (height, width).
        Values peak at 1.0 in the center and smoothly taper towards edges.
        """
        # 1D Hann windows
        w_y = np.hanning(height)
        w_x = np.hanning(width)

        # Outer product to form 2D window
        window_2d = np.outer(w_y, w_x).astype(np.float32)

        # Clamp minimum weight to prevent numerical instability at edges
        window_2d = np.maximum(window_2d, min_weight)
        return window_2d

    @staticmethod
    def get_cosine_window_2d(height: int, width: int, min_weight: float = 1e-4) -> np.ndarray:
        """Generates a 2D Cosine window of shape (height, width)."""
        y = np.linspace(-np.pi / 2, np.pi / 2, height)
        x = np.linspace(-np.pi / 2, np.pi / 2, width)
        w_y = np.cos(y)
        w_x = np.cos(x)
        window_2d = np.outer(w_y, w_x).astype(np.float32)
        window_2d = np.maximum(window_2d, min_weight)
        return window_2d
