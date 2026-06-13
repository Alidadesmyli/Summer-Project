"""
XAI explanation generators.
- GradCAM / GradCAM++ for DenseNet-121 (CNN)
- Attention Rollout for ViT-B/16 (Abnar & Zuidema, 2020)
Week 5 deliverable.
"""
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_gradcam_explanation(
    model: torch.nn.Module,
    xrv_arr: np.ndarray,
    target_class: int,
    device: torch.device,
    use_plusplus: bool = False,
) -> np.ndarray:
    """
    Returns a (224, 224) GradCAM heatmap in [0, 1] for DenseNet-121.

    Args:
        model:        torchxrayvision DenseNet loaded on `device`
        xrv_arr:      (1, 224, 224) float array normalised for torchxrayvision
        target_class: index into model.pathologies
        device:       torch device
        use_plusplus: use GradCAM++ instead of GradCAM
    """
    target_layer = list(model.features.denseblock4.children())[-1].conv2
    tensor = torch.from_numpy(xrv_arr).unsqueeze(0).to(device)  # (1,1,224,224)
    cam_cls = GradCAMPlusPlus if use_plusplus else GradCAM
    with cam_cls(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        heatmap = cam(input_tensor=tensor, targets=targets)[0]   # (224, 224)
    return heatmap


def get_attention_rollout(
    model: torch.nn.Module,
    processor,
    pil_rgb: Image.Image,
    device: torch.device,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Returns a (224, 224) Attention Rollout map in [0, 1] for ViT-B/16.

    Propagates attention through all transformer layers by multiplying
    attention matrices with skip connections (Abnar & Zuidema, 2020).

    Args:
        model:        HuggingFace ViTForImageClassification on `device`
        processor:    corresponding ViTImageProcessor
        pil_rgb:      RGB PIL image (224×224)
        device:       torch device
        discard_ratio: fraction of lowest-attention tokens zeroed per layer
    """
    inputs = processor(images=pil_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (1, heads, seq_len, seq_len)

    result = torch.eye(attentions[0].size(-1), device=device)
    for attn in attentions:
        attn_avg = attn.mean(dim=1).squeeze(0)                          # (197, 197)
        threshold = torch.quantile(attn_avg.flatten(), discard_ratio)
        attn_avg = torch.where(attn_avg < threshold, torch.zeros_like(attn_avg), attn_avg)
        attn_avg = attn_avg + torch.eye(attn_avg.size(0), device=device)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = attn_avg @ result

    # CLS token (row 0) attention over patch tokens (columns 1:)
    mask = result[0, 1:].cpu().numpy()      # (196,)
    side = int(np.sqrt(mask.size))          # 14
    mask = mask.reshape(side, side)

    mask_img = Image.fromarray((mask / (mask.max() + 1e-8) * 255).astype(np.uint8)).resize(
        (224, 224), Image.BILINEAR
    )
    return np.array(mask_img, dtype=np.float32) / 255.0
