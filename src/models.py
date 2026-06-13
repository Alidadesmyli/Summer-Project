"""
Model loading utilities for DenseNet-121 (torchxrayvision) and ViT-B/16 (HuggingFace).
Week 4 deliverable.
"""
import torch
import torchxrayvision as xrv
from transformers import ViTForImageClassification, ViTImageProcessor


CHEXPERT_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Pleural Effusion"
]


def load_densenet(device: torch.device) -> torch.nn.Module:
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()
    return model.to(device)


def load_vit(device: torch.device):
    checkpoint = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(CHEXPERT_PATHOLOGIES),
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return model.to(device), processor


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
