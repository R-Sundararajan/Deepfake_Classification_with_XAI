"""
Attention Rollout Visualization
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms

from models.vit import VisionTransformer
from explainability.attention_rollout import AttentionRollout

from utils.config import (
    DEVICE,
    CHECKPOINT_DIR,
    IMAGE_SIZE,
)


# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])



# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = VisionTransformer().to(DEVICE)

checkpoint = CHECKPOINT_DIR / "best_vit.pth"


# --------------------------------------------------
# Visualize Function
# --------------------------------------------------

def visualize(image_path, save_path=None):

    print(f"Visualizing: {image_path}")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
        )
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probability,
            dim=1,
        )

        fake_probability = probability[0, 0].item()
        real_probability = probability[0, 1].item()

    rollout = AttentionRollout(model)

    original = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

    overlay = rollout.overlay(original)

    label = "Real" if prediction.item() == 1 else "Fake"

    confidence_pct = confidence.item() * 100

    print(f"Prediction : {label}")
    print(f"Fake probability : {fake_probability * 100:.2f}%")
    print(f"Real probability : {real_probability * 100:.2f}%")

    # ------------------------------------------
    # Display / Save
    # ------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(original)

    plt.title("Original Image")

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(overlay)

    plt.title(
        f"{label}\nConfidence : {confidence_pct:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    if save_path:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.show()
