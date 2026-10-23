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


def current_image_candidates():
    """Resolve one atomic, hash-bound catalog without rewriting historical runs.

    A malformed new catalog fails closed to the last verified single-model
    pointer. Model IDs are aliases for existing records, never extra variants.
    """
    fallback = active_image_candidate()
    pointer = PROJECT_ROOT/'configs/runtime/current_image_models.json'
    if not pointer.is_file():
        return [fallback] if fallback else []
    try:
        activation = json.loads(pointer.read_text(encoding='utf-8'))
        if activation['format'] != 'current_image_models_pointer_v1':
            raise ValueError('Unsupported current image pointer')
        summary = Path(activation['summary']).resolve()
        if not summary.is_relative_to(PROJECT_ROOT/'results') or checkpoint_digest(summary) != activation['summary_sha256']:
            raise ValueError('Current catalog digest mismatch')
        deployment = json.loads(summary.read_text(encoding='utf-8'))
        records = deployment['records']
        if not deployment['verified'] or not deployment['gates'] or not all(deployment['gates'].values()):
            raise ValueError('Current image deployment is unverified')
        if {r['id'] for r in records} != {'R18_FT_20261021', 'E02_efficientnet_b0'} or len(records) != 2:
            raise ValueError('Only the two existing logical identities may be updated')
        for record in records:
            checkpoint = Path(record['checkpoint']).resolve()
            digest = record['checkpoint_sha256']
            if not checkpoint.is_relative_to(PROJECT_ROOT/'results') or checkpoint_digest(checkpoint) != digest:
                raise ValueError('Current checkpoint digest mismatch')
            if record['source'] != 'image_continuation' or record['evaluation_unit'] != 'image':
                raise ValueError('Current pointer is image-only')
            for key in ['validation', 'full_validation', 'held_out_test']:
                metrics = record[key]
                if not metrics['completed'] or metrics['checkpoint_sha256'] != digest:
                    raise ValueError('Metrics are not bound to the selected checkpoint')
            if record['full_validation']['samples'] != 20000 or record['held_out_test']['samples'] != 20000:
                raise ValueError('Full evaluation evidence is required')
        return copy.deepcopy(records)
    except (OSError, ValueError, TypeError, KeyError):
        return [fallback] if fallback else []
