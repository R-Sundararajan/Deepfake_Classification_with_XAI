"""Compare real checkpoint maps from concurrent clones with serial execution."""
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from config import PROJECT_ROOT
from explainability.suite import ParallelExplainer, DEFAULT_PARAMETERS
from inference.catalog import catalog, create_predictor, methods_for
from tools.runtime import configure_runtime
from training.video_protocol import RUN


def run(output):
    configure_runtime()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with (RUN / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    clip = sorted([row for row in rows if row['split'] == 'validation' and row['label'] == 'FAKE'], key=lambda row: row['filename'])[0]
    cases = [('Image', 'S0', Path(r'C:\Users\Rangarajan\OneDrive\Pictures\Saved Pictures\random-person.jpeg')),
             ('Video', 'S8', Path(clip['path']))]
    report = []
    for modality, identifier, path in cases:
        record = next(row for row in catalog(modality) if row['id'] == identifier)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        predictor = create_predictor(record)
        if modality == 'Image':
            with Image.open(path) as image:
                result = predictor.predict_image(image, method=None)
            positions = ()
        else:
            result = predictor.predict_video(path, method=None)
            positions = (int(np.argmax([row['fake_probability'] for row in result['sampled_frames']])),)
        score = result['fake_probability']
        runner = ParallelExplainer()
        tick = time.perf_counter()
        try:
            suite = runner.start(record, result, path, modality, positions, predictor=predictor)
            parallel = {method: future.result(timeout=180) for method, future in suite.futures.items()}
            parallel_seconds = time.perf_counter() - tick
            checks = []
            tick = time.perf_counter()
            for method in methods_for(record):
                parameters = dict(DEFAULT_PARAMETERS.get(method, {}))
                if modality == 'Video':
                    parameters['frame_indices'] = list(positions)
                serial = predictor.explain_result(result, method, result['prediction'], **parameters)
                before = parallel[method][0] if modality == 'Video' else parallel[method]
                after = serial[0] if modality == 'Video' else serial
                maximum_error = float(np.abs(before['cam'] - after['cam']).max())
                np.testing.assert_allclose(before['cam'], after['cam'], atol=1e-4, rtol=1e-4)
                assert before['details']['target_class'] == result['prediction']
                checks.append({'method': method, 'max_absolute_map_difference': maximum_error,
                               'execution': before['details']['execution']})
            serial_seconds = time.perf_counter() - tick
            modules = ([predictor.legacy] if identifier == 'S0' else predictor.encoders + [predictor.spatial_head, predictor.temporal_head])
            assert all(not module._forward_hooks for root in modules if root is not None for module in root.modules())
            for encoder in getattr(predictor, 'encoders', []):
                if hasattr(encoder, 'blocks'):
                    assert all(block.attn.fused_attn for block in encoder.blocks)
            if modality == 'Image':
                with Image.open(path) as image:
                    repeated = predictor.predict_image(image, method=None)
            else:
                repeated = predictor.predict_video(path, method=None)
            assert abs(repeated['fake_probability'] - score) < 1e-7
            item = {'model_id': identifier, 'modality': modality, 'prediction': result['prediction'],
                    'fake_probability_before': score, 'fake_probability_after': repeated['fake_probability'],
                    'parallel_seconds_including_clone': parallel_seconds,
                    'serial_seconds_warm_model': serial_seconds, 'methods': checks,
                    'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                    'peak_reserved_cuda_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
            report.append(item)
            print(f'{modality} {identifier}: {len(checks)} concurrent/serial maps match; prediction unchanged.', flush=True)
        finally:
            runner.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/demo_presentation_20261020/parallel_parity.json')
    run(parser.parse_args().output)
