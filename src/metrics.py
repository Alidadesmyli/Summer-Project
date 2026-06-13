"""
Explanation stability metrics (Week 7 deliverable).

  - SSIM: structural similarity between two heatmaps
  - IoU:  intersection-over-union of top-k activated pixels (k = top 20%)
"""
import numpy as np
from skimage.metrics import structural_similarity as ssim


def measure_stability(
    exp_original: np.ndarray,
    exp_perturbed: np.ndarray,
    top_k_fraction: float = 0.20,
) -> dict[str, float]:
    """
    Compute SSIM and top-k IoU between two explanation heatmaps.

    Args:
        exp_original:   (H, W) float array in [0, 1]
        exp_perturbed:  (H, W) float array in [0, 1], same shape
        top_k_fraction: fraction of pixels to treat as 'activated' (default 20%)

    Returns:
        {'ssim': float, 'iou': float}  both in [0, 1]
    """
    assert exp_original.shape == exp_perturbed.shape, "Heatmap shapes must match"

    ssim_score = float(ssim(exp_original, exp_perturbed, data_range=1.0))

    k = max(1, int(exp_original.size * top_k_fraction))
    flat_orig = exp_original.flatten()
    flat_pert = exp_perturbed.flatten()

    top_orig = set(np.argpartition(flat_orig, -k)[-k:])
    top_pert = set(np.argpartition(flat_pert, -k)[-k:])

    intersection = len(top_orig & top_pert)
    union = len(top_orig | top_pert)
    iou_score = intersection / union if union > 0 else 0.0

    return {"ssim": ssim_score, "iou": iou_score}
