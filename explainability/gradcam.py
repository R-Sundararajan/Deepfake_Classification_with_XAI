"""Small, dependency-light Grad-CAM implementation for CNN classifiers."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.details = {}

    def _save_activation(self, _module, _inputs, output):
        self.activations = output

    def generate(self, input_tensor: torch.Tensor, class_index: int, method: str = 'gradcam') -> np.ndarray:
        if method not in {'gradcam', 'gradcam++', 'layercam'}:
            raise ValueError('Unsupported CAM method')
        self.activations = self.gradients = None
        with torch.enable_grad():
            input_tensor = input_tensor.detach().clone().requires_grad_(True)
            logits = self.model(input_tensor)
            if self.activations is None or self.activations.ndim != 4:
                raise RuntimeError('CAM requires a convolutional BCHW activation.')
            self.gradients = torch.autograd.grad(logits[:, class_index].sum(), self.activations)[0].detach()
        self.activations = self.activations.detach()
        if not torch.isfinite(self.gradients).all():
            raise ValueError('Nonfinite CAM gradients')

        if method == 'gradcam++':
            # Common first-gradient power approximation to Grad-CAM++ weights.
            squared = self.gradients.square()
            denominator = 2*squared + self.activations.sum((2,3),keepdim=True)*self.gradients.pow(3)
            safe = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
            alpha = torch.where(denominator != 0, squared/safe, torch.zeros_like(squared))
            weights = (alpha*torch.relu(self.gradients)).sum((2,3),keepdim=True)
        elif method == 'layercam':
            weights = torch.relu(self.gradients)
        else:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        maximum = float(cam.max())
        self.details = {'activation_shape': list(self.activations.shape), 'gradient_shape': list(self.gradients.shape),
                        'gradient_abs_sum': float(self.gradients.abs().sum()), 'raw_map_max': maximum,
                        'raw_map_min': float(cam.min()), 'low_signal': maximum <= 1e-12, 'target_index': int(class_index),
                        'display': 'Relative positive influence on the selected class score'}
        if maximum > 1e-12:
            cam /= maximum
        else:
            cam.zero_()
        return cam.cpu().numpy()

    def close(self) -> None:
        self._forward_handle.remove()
        self.activations = self.gradients = None


def create_visualizations(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    overlay_strength: float = 0.45,
    signed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= overlay_strength <= 1:
        raise ValueError('Overlay strength must be between zero and one')
    if not np.isfinite(cam).all():
        raise ValueError('Nonfinite display map')
    height, width = original_rgb.shape[:2]
    resized_cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    if signed:
        value = np.clip(resized_cam, -1, 1)
        positive, negative = np.maximum(value, 0), np.maximum(-value, 0)
        heatmap_rgb = np.uint8(np.clip(np.stack([1-negative, 1-positive-negative, 1-positive], -1)*255, 0, 255))
    else:
        value = np.clip(resized_cam, 0, 1)
        heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*value), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    # Zero attribution must preserve the original, instead of tinting it blue.
    alpha = (overlay_strength*np.abs(value))[..., None]
    overlay = np.uint8(np.clip(np.asarray(original_rgb)*(1-alpha)+heatmap_rgb*alpha, 0, 255))
    return heatmap_rgb, overlay


def describe_regions(cam: np.ndarray) -> str:
    """Describe coarse spatial activation without claiming semantic artifacts."""
    row_names = ("upper", "central", "lower")
    column_names = ("left", "central", "right")
    row_edges = np.linspace(0, cam.shape[0], 4, dtype=int)
    column_edges = np.linspace(0, cam.shape[1], 4, dtype=int)
    regions = []
    for row in range(3):
        for column in range(3):
            region = cam[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            if row == 1 and column == 1:
                name = "central"
            else:
                name = f"{row_names[row]}-{column_names[column]}"
            regions.append((float(region.mean()), name))
    regions.sort(reverse=True)
    selected = [regions[0][1]]
    if regions[1][0] >= regions[0][0] * 0.9:
        selected.append(regions[1][1])
    return " and ".join(selected)
