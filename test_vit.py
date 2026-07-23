import torch

from models.vit import VisionTransformer

model = VisionTransformer()

dummy = torch.randn(2, 3, 256, 256)

output = model(dummy)

print("=" * 50)
print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)
print(output)
print("=" * 50)