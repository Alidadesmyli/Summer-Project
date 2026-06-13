"""
XAI explanation generators.
- GradCAM / GradCAM++ for DenseNet-121 (CNN)
- Attention Rollout for ViT-B/16
Week 5 deliverable.
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_gradcam_explanation(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    use_plusplus: bool = False,
) -> np.ndarray:
    """Returns a (H, W) heatmap in [0, 1] for the given class."""
    target_layer = model.features.denseblock4.denselayer16.conv2
    cam_cls = GradCAMPlusPlus if use_plusplus else GradCAM
    with cam_cls(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    return grayscale_cam[0]


def get_attention_rollout(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """Returns a (H, W) attention rollout map in [0, 1] for ViT-B/16."""
    from pytorch_grad_cam import AblationCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # pytorch-grad-cam provides VitAttentionGradRollout for ViT attention rollout
    from pytorch_grad_cam.vit_gradcam import VitAttentionRollout

    rollout = VitAttentionRollout(model, discard_ratio=discard_ratio)
    mask = rollout(image_tensor)
    return mask
