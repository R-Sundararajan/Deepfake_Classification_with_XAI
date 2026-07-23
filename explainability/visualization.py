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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = VisionTransformer().to(DEVICE)

checkpoint = CHECKPOINT_DIR / "best_vit.pth"

print("Checkpoint Used:")
print(checkpoint.resolve())

model.load_state_dict(
    torch.load(
        checkpoint,
        map_location=DEVICE,
    )
)

model.eval()


# --------------------------------------------------
# Visualize Function
# --------------------------------------------------

def visualize(image_path):

    print(repr(image_path))
    print(type(image_path))

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    print("Minimum:", input_tensor.min())
    print("Maximum:", input_tensor.max())
    
    with torch.no_grad():

        output = model(input_tensor)

        print(output)

        probability = torch.softmax(output, dim=1)

        print(probability)

        confidence, prediction = torch.max(
            probability,
            dim=1,
        )

        print(prediction)

    rollout = AttentionRollout(model)

    original = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

    overlay = rollout.overlay(original)

    label = "Real" if prediction.item() == 1 else "Fake"

    confidence = confidence.item() * 100

    # ------------------------------------------
    # Display
    # ------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(original)

    plt.title("Original Image")

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    ))

    plt.title(
        f"{label}\nConfidence : {confidence:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()