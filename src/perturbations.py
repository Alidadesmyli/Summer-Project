"""
Clinically plausible image perturbations (Week 6 deliverable).

Perturbation types and levels from the project spec:
  - gaussian_noise:  sigma in {0.05, 0.10, 0.20}
  - contrast_shift:  factor in {0.70, 0.85, 1.15, 1.30}
  - rotation:        degrees in {5, 10, 15}
"""
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

PERTURBATION_LEVELS: dict[str, list] = {
    "gaussian_noise": [0.05, 0.10, 0.20],
    "contrast_shift": [0.70, 0.85, 1.15, 1.30],
    "rotation":       [5,    10,   15],
}


def apply_perturbation(image: Image.Image, kind: str, level: float) -> Image.Image:
    """
    Apply a single perturbation to a PIL image.

    Args:
        image: input PIL image (grayscale or RGB)
        kind:  one of 'gaussian_noise', 'contrast_shift', 'rotation'
        level: intensity value from PERTURBATION_LEVELS[kind]

    Returns:
        Perturbed PIL image of the same size and mode.
    """
    if kind == "gaussian_noise":
        arr = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, level, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8), mode=image.mode)

    if kind == "contrast_shift":
        return TF.adjust_contrast(image, contrast_factor=level)

    if kind == "rotation":
        return TF.rotate(image, angle=level, expand=False)

    raise ValueError(f"Unknown perturbation kind: {kind!r}")
