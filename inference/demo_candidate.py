"""Fail-closed, reversible pointer for a fully verified image-only candidate."""
import copy
import json
from pathlib import Path

from config import PROJECT_ROOT
from inference.contracts import checkpoint_digest


def active_image_candidate():
    pointer = PROJECT_ROOT/'configs/runtime/active_image_model.json'
    if not pointer.is_file():
        return None
    try:
        activation = json.loads(pointer.read_text(encoding='utf-8'))
        if activation['format'] != 'demo_image_pointer_v1':
            return None
        summary = Path(activation['summary']).resolve()
        if not summary.is_relative_to(PROJECT_ROOT/'results') or checkpoint_digest(summary) != activation['summary_sha256']:
            return None
        deployment = json.loads(summary.read_text(encoding='utf-8'))
        if not deployment['verified'] or not all(deployment['gates'].values()):
            return None
        record = deployment['record']
        checkpoint = Path(record['checkpoint']).resolve()
        if not checkpoint.is_relative_to(PROJECT_ROOT/'results') or checkpoint_digest(checkpoint) != record['checkpoint_sha256']:
            return None
        if record['evaluation_unit'] != 'image' or record['source'] != 'image_continuation':
            return None
        if record['full_validation']['checkpoint_sha256'] != record['checkpoint_sha256'] or record['held_out_test']['checkpoint_sha256'] != record['checkpoint_sha256']:
            return None
        if not record['full_validation']['completed'] or not record['held_out_test']['completed']:
            return None
        return copy.deepcopy(record)
    except (OSError, ValueError, TypeError, KeyError):
        return None
