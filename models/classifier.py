"""ResNet-18 transfer-learning classifier and checkpoint helpers."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from config import NUM_CLASSES, TRAINABLE_BACKBONE_BLOCKS


def create_model(
    pretrained: bool = True,
    trainable_backbone_blocks: int = TRAINABLE_BACKBONE_BLOCKS,
) -> nn.Module:
    """Create a binary ResNet-18 and fine-tune only its final residual blocks."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    if trainable_backbone_blocks:
        for block in blocks[-trainable_backbone_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def target_layer(model: nn.Module) -> nn.Module:
    """Return the final convolution used by Grad-CAM."""
    return model.layer4[-1].conv2


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trainable_blocks = int(checkpoint.get("trainable_backbone_blocks", 1))
    model = create_model(pretrained=False, trainable_backbone_blocks=trainable_blocks)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
