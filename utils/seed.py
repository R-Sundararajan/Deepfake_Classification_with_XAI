"""
Random Seed Configuration

This module ensures reproducibility by fixing the random seed
across Python, NumPy, and PyTorch.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False