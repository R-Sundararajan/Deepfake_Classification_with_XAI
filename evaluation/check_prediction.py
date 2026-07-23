import random
import torch
from torchvision import datasets, transforms

from models.vit import VisionTransformer
from utils.config import *

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

model = VisionTransformer().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT_DIR/"best_vit.pth",
        map_location=DEVICE
    )
)

model.eval()

indices = random.sample(range(len(dataset)), 10)

print()

for idx in indices:

    image, label = dataset[idx]

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        prob = torch.softmax(output, dim=1)

        confidence, pred = torch.max(prob, dim=1)

    actual = "Fake" if label == 0 else "Real"
    predicted = "Fake" if pred.item() == 0 else "Real"

    print(
        f"Actual: {actual:5} | "
        f"Predicted: {predicted:5} | "
        f"Confidence: {confidence.item()*100:.2f}%"
    )