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
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        logits[:, class_index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        return cam.cpu().numpy()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def create_visualizations(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    overlay_strength: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = original_rgb.shape[:2]
    resized_cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        original_rgb.astype(np.uint8),
        1.0 - overlay_strength,
        heatmap_rgb,
        overlay_strength,
        0,
    )
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
