"""Explicit class mapping, checkpoint fingerprints and synchronized stage timings."""
from functools import lru_cache
from pathlib import Path
import math
import time
import torch
from tools.acquire_dfdc import sha256


def class_probabilities(logits, class_to_idx, temperature=1., threshold=.5):
    if class_to_idx not in ({'real': 0, 'fake': 1}, {'fake': 0, 'real': 1}):
        raise ValueError('Unsupported binary class mapping')
    if logits.ndim != 2 or logits.shape[-1] != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite two-logit outputs')
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= threshold <= 1:
        raise ValueError('Invalid calibration or threshold')
    p = (logits.float()/temperature).softmax(-1)[:, class_to_idx['fake']]
    return {'fake': p, 'real': 1-p, 'predicted_fake': p >= threshold,
            'predicted_class_score': torch.where(p >= threshold, p, 1-p)}


@lru_cache(maxsize=128)
def _digest(path, size, mtime_ns):
    return sha256(path)


def checkpoint_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def measure(device, call):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = call()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result, (time.perf_counter()-start)*1000
