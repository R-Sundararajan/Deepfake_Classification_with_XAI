"""
Vision Transformer (ViT)

Complete Vision Transformer
"""

import torch
import torch.nn as nn

from utils.config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    NUM_CLASSES,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    MLP_DIM,
    DROPOUT,
)
# --------------------------------------------------
# Patch Embedding
# --------------------------------------------------

class PatchEmbedding(nn.Module):

    def __init__(
    self,
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=3,
    embed_dim=EMBED_DIM,
):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x):

        B = x.shape[0]

        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls, x), dim=1)

        x = x + self.position_embedding

        return x


# --------------------------------------------------
# Multi Head Self Attention
# --------------------------------------------------

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.projection = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.attetion_map = None

    def forward(self, x):

        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attention = self.softmax(attention)

        self.attention_map = attention.detach()

        out = attention @ v

        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.projection(out)

        return out


# --------------------------------------------------
# Feed Forward Network
# --------------------------------------------------

class MLP(nn.Module):

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),

        )

    def forward(self, x):

        return self.network(x)


# --------------------------------------------------
# Transformer Encoder
# --------------------------------------------------

class TransformerEncoder(nn.Module):

    def __init__(
        self,
       embed_dim=EMBED_DIM,
       num_heads=NUM_HEADS,
       mlp_dim=MLP_DIM,
       dropout=DROPOUT,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            embed_dim,
            mlp_dim,
            dropout,
        )

    def forward(self, x):

        x = x + self.attention(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x


# --------------------------------------------------
# Vision Transformer
# --------------------------------------------------

class VisionTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
        )

        self.encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embed_dim=EMBED_DIM,
                    num_heads=NUM_HEADS,
                    mlp_dim=MLP_DIM,
                    dropout=DROPOUT,
                )
                for _ in range(NUM_LAYERS)
            ]
        )

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.head = nn.Linear(
            EMBED_DIM,
            NUM_CLASSES,
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        x = self.encoder(x)

        x = self.norm(x)

        cls_token = x[:, 0]

        output = self.head(cls_token)

        return output