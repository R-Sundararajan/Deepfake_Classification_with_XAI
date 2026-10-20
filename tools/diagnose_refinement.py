"""Reproduce the supplied sanity images without using them for model selection.

Run before/after repairs with different, new output directories. Historical
artifacts and the frozen DFDC test are never rewritten by this command.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config import PROJECT_ROOT
from inference.media import MediaPredictor
from inference.unified import ResearchPredictor, available_models
from tools.acquire_dfdc import sha256
from tools.runtime import configure_runtime


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call):
    synchronize()
    start = time.perf_counter()
    value = call()
    synchronize()
    return value, (time.perf_counter() - start) * 1000


def face_coordinates(cropper, rgb):
    """Record geometry in original pixels, including alignment rather than a fake bbox crop."""
    height, width = rgb.shape[:2]
    if not hasattr(cropper, 'detector'):
        return {'bounds_xyxy': [0, 0, width, height], 'face_detection_requested': False}
    scale = min(1., 640 / max(height, width))
    small = cv2.resize(rgb[:, :, ::-1].copy(), (max(1, round(width * scale)), max(1, round(height * scale))))
    cropper.detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = cropper.detector.detect(small)
    if faces is None or not len(faces):
        return {'bounds_xyxy': [0, 0, width, height], 'face_detection_requested': True, 'detected': False}
    face = sorted(faces, key=lambda row: (-float(row[2] * row[3]), float(row[0]), float(row[1])))[0].copy()
    face[:14] /= scale
    x, y, w, h = (float(value) for value in face[:4])
    result = {'face_detection_requested': True, 'detected': True,
              'detector_bbox_xyxy': [x, y, x + w, y + h], 'detector_confidence': float(face[-1]),
              'landmarks_xy': face[4:14].reshape(5, 2).tolist(), 'margin': cropper.margin}
    if cropper.align:
        points = face[4:10].reshape(3, 2).astype(np.float32)
        target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366]], np.float32) * 2
        matrix = cv2.getAffineTransform(points, (target - 112) / cropper.margin + 112)
        inverse = cv2.invertAffineTransform(matrix)
        corners = np.array([[0, 0, 1], [223, 0, 1], [223, 223, 1], [0, 223, 1]], np.float64)
        result.update(affine_original_to_224=matrix.tolist(), input_corners_original_xy=(corners @ inverse.T).tolist())
    return result


def tensor_details(value):
    x = value.detach().float().cpu()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device),
            'finite': bool(torch.isfinite(x).all()), 'min': float(x.min()), 'max': float(x.max()),
            'channel_mean': x.mean((0, 2, 3)).tolist(), 'channel_std': x.std((0, 2, 3)).tolist(),
            'color': 'RGB', 'normalization_mean': [.485, .456, .406], 'normalization_std': [.229, .224, .225]}


def run(output, inputs):
    configure_runtime()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    for relative in ['app.py', 'inference/unified.py', 'inference/media.py', 'explainability/gradcam.py', 'explainability/methods.py']:
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, target)
    records = available_models()
    candidates = [(record['id'], record['checkpoint'], record['config']['model'],
                   'Real vs Fake Faces' if record['id'] == 'S0' else 'Local source-grouped DFDC') for record in records]
    for summary in sorted((PROJECT_ROOT / 'results/execution_20261019').glob('*/summary.json')):
        record = json.loads(summary.read_text(encoding='utf-8'))
        checkpoint = record.get('checkpoint')
        if checkpoint and Path(checkpoint).is_file() and all(str(Path(checkpoint)) != str(Path(row[1])) for row in candidates):
            candidates.append((summary.parent.name, checkpoint, str(record.get('model', 'image transfer')), record.get('dataset', 'Real vs Fake Faces')))
    rows = []
    for identifier, checkpoint, architecture, dataset in candidates:
        predictor, load_ms = timed(lambda: ResearchPredictor(identifier) if identifier.startswith('S') else MediaPredictor(checkpoint))
        for path, expected in inputs:
            path = Path(path)
            rgb = np.asarray(Image.open(path).convert('RGB'))
            geometry = face_coordinates(predictor.crop, rgb)
            method = 'gradcam' if not hasattr(predictor, 'methods') or 'gradcam' in predictor.methods else 'integrated_gradients'
            original_explain = predictor.explain
            explanation_times = []
            def explain(*args, **kwargs):
                value, elapsed = timed(lambda: original_explain(*args, **kwargs))
                explanation_times.append(elapsed)
                return value
            predictor.explain = explain
            try:
                result, total_ms = timed(lambda: predictor.predict_image(Image.fromarray(rgb), method))
                # Same warm, resident model/input; report a real repeated call too.
                _, warm_ms = timed(lambda: predictor.predict_image(Image.fromarray(rgb), method))
            finally:
                predictor.explain = original_explain
            example_dir = output / identifier / path.stem
            example_dir.mkdir(parents=True)
            explanation = result['explanation']
            for key in ['original', 'heatmap', 'overlay']:
                Image.fromarray(np.asarray(explanation[key])).save(example_dir / (key + '.png'))
            np.save(example_dir / 'display_map.npy', explanation['cam'])
            array = np.asarray(explanation['original'])
            tensor = predictor.tensor(array[None]) if hasattr(predictor, 'tensor') else predictor.transform(Image.fromarray(array)).unsqueeze(0).to(predictor.device)
            details = tensor_details(tensor)
            np.save(example_dir / 'model_tensor.npy', tensor.detach().cpu().numpy())
            row = {'id': identifier, 'architecture': architecture, 'training_dataset': dataset,
                   'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                   'input': str(path), 'input_sha256': sha256(path), 'expected': expected,
                   **{key: result[key] for key in ['prediction', 'fake_probability', 'real_probability', 'confidence', 'crop_method']},
                   'correct_sanity_label': result['prediction'] == expected, 'geometry': geometry,
                   'model_input': details, 'threshold': .5, 'temperature': 1., 'calibration': 'uncalibrated image score',
                   'checkpoint_load_ms': load_ms, 'total_first_ms': total_ms, 'total_warm_ms': warm_ms,
                   'explanation_first_ms': explanation_times[0], 'explanation_warm_ms': explanation_times[1],
                   'method': method, 'explanation_details': explanation.get('details', {}),
                   'crop_path': str(example_dir / 'original.png'), 'image_fallback': result.get('image_fallback')}
            if identifier in {'S0', 'S2'}:
                # Preserve raw old IG diagnostics before changing display/integration.
                target = int(result['fake_probability'] >= .5)
                extra, elapsed = timed(lambda: original_explain(Image.fromarray(array), target, 'integrated_gradients', image_input=True))
                row['ig'] = {'milliseconds': elapsed, 'details': extra.get('details', {})}
                Image.fromarray(extra['overlay']).save(example_dir / 'integrated_gradients_overlay.png')
            rows.append(row)
            (output / 'diagnostics.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
            print(f"{identifier} {path.name}: {result['prediction']} FAKE={result['fake_probability']:.6f}; {total_ms:.1f} ms, warm {warm_ms:.1f} ms", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    lines = ['# Supplied-image diagnostics', '',
             'User-provided sanity labels; these two images are not a benchmark or training/selection data.', '',
             '| ID | Input | Expected | Predicted | FAKE | REAL | Crop | First / warm ms | XAI ms |',
             '|---|---|---|---|---:|---:|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['id']} | {Path(row['input']).name} | {row['expected']} | {row['prediction']} | {row['fake_probability']:.2%} | {row['real_probability']:.2%} | {row['crop_method']} | {row['total_first_ms']:.1f} / {row['total_warm_ms']:.1f} | {row['explanation_first_ms']:.1f} |")
    lines += ['', 'Full checkpoint hashes, crop geometry, actual normalized tensors, calibration and timing records: `diagnostics.json`.',
              'Each input directory contains its actual model crop, tensor array, heatmap and overlay.',
              'Smoke checkpoints, where present, are diagnostic artifacts and do not enter model leaderboards.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=Path, default=Path(r'C:\Users\Rangarajan\OneDrive\Pictures\Saved Pictures'))
    args = parser.parse_args()
    run(args.output, [(args.images / 'random-person.jpeg', 'FAKE'), (args.images / 'real.jpeg', 'REAL')])
