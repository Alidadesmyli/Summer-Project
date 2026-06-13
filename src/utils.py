"""
Shared utilities: image loading, heatmap overlay, results I/O.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


RESULTS_COLUMNS = [
    "image_id",
    "architecture",
    "perturbation_type",
    "perturbation_level",
    "ssim",
    "iou",
]


def load_image(path: str | Path, size: int = 224) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((size, size))


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a (H,W) heatmap over an (H,W,3) image. Returns (H,W,3) uint8."""
    cmap = plt.get_cmap(colormap)
    heat_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heat_rgb + (1 - alpha) * image).astype(np.uint8)
    return blended


def append_result(
    results: list[dict],
    image_id: str,
    architecture: str,
    perturbation_type: str,
    perturbation_level: float,
    ssim: float,
    iou: float,
) -> None:
    results.append({
        "image_id": image_id,
        "architecture": architecture,
        "perturbation_type": perturbation_type,
        "perturbation_level": perturbation_level,
        "ssim": ssim,
        "iou": iou,
    })


def save_results(results: list[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(results, columns=RESULTS_COLUMNS)
    df.to_csv(path, index=False)
    return df
