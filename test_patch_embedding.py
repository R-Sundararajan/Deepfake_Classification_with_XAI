import torch

from models.vit import PatchEmbedding
from models.vit import TransformerEncoder

patch = PatchEmbedding()

encoder = TransformerEncoder()

dummy = torch.randn(2, 3, 256, 256)

x = patch(dummy)

print("Patch Output   :", x.shape)

x = encoder(x)

print("Encoder Output :", x.shape)