"""
Attention Rollout for Vision Transformer
"""

import cv2
import numpy as np
import torch


class AttentionRollout:

    def __init__(self, model):

        self.model = model

    # --------------------------------------------------
    # Compute Attention Rollout
    # --------------------------------------------------

    def rollout(self):

        attention_matrices = []

        # Collect attention maps from every encoder block
        for block in self.model.encoder:

            attention = block.attention.attention_map

            if attention is None:
                raise RuntimeError(
                    "Attention map not found. "
                    "Run a forward pass before calling AttentionRollout."
                )

            # Shape:
            # (Batch, Heads, Tokens, Tokens)

            attention = attention.mean(dim=1)

            attention_matrices.append(attention)

        num_tokens = attention_matrices[0].size(-1)

        result = torch.eye(
            num_tokens,
            device=attention_matrices[0].device,
        ).unsqueeze(0)

        for attention in attention_matrices:

            identity = torch.eye(
                num_tokens,
                device=attention.device,
            ).unsqueeze(0)

            attention = attention + identity

            attention = attention / attention.sum(
                dim=-1,
                keepdim=True,
            )

            result = torch.bmm(attention, result)

        return result

    # --------------------------------------------------
    # Generate Attention Mask
    # --------------------------------------------------

    def generate_mask(self):

        rollout = self.rollout()

        mask = rollout[0, 0, 1:]

        num_patches = int(np.sqrt(mask.shape[0]))

        mask = mask.reshape(
            num_patches,
            num_patches,
        )

        mask = mask.cpu().numpy()

        mask -= mask.min()

        mask /= (mask.max() + 1e-8)

        return mask

    # --------------------------------------------------
    # Overlay Heatmap
    # --------------------------------------------------

    def overlay(self, image):

        """
        image:
            RGB image
            shape (H,W,3)
        """

        mask = self.generate_mask()

        heatmap = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB,
        )

        overlay = (
            0.6 * image +
            0.4 * heatmap
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        return overlay