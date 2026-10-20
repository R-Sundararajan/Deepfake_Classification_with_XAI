"""Actual-checkpoint XAI, video and profile checks; outputs use a new namespace."""
import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from inference.catalog import catalog, create_predictor
from explainability.methods import check_attribution
from tools.diagnose_refinement import timed
from tools.runtime import configure_runtime
from training.video_protocol import RUN, normalized_indices

ROOT = RUN.parent/'debug_refinement_20261019'
PICTURES = Path(r'C:\Users\Rangarajan\OneDrive\Pictures\Saved Pictures')


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def image_checks(output):
    output.mkdir(parents=True, exist_ok=False)
    records = catalog('Video')+[record for record in catalog('Image') if record['id'] != 'S0']
    rows = []
    for record in records:
        predictor = create_predictor(record)
        for name in ['random-person.jpeg', 'real.jpeg']:
            with Image.open(PICTURES/name) as image:
                result = predictor.predict_image(image, method=None)
            array = result['prepared_input']
            tensor = predictor.tensor(array[None]) if hasattr(predictor, 'tensor') else predictor.transform(Image.fromarray(array)).unsqueeze(0).to(predictor.device)
            view = predictor.attribution_model(image=True) if hasattr(predictor, 'attribution_model') else predictor.model
            class_mapping = {'REAL': 0, 'FAKE': 1} if hasattr(predictor, 'attribution_model') else {'FAKE': 0, 'REAL': 1}
            with torch.inference_mode():
                reference = view(tensor).float().softmax(-1).cpu()
            for method in predictor.methods:
                for target in [result['prediction'], 'REAL' if result['prediction'] == 'FAKE' else 'FAKE']:
                    explanation, milliseconds = timed(lambda: predictor.explain_result(result, method, target))
                    cam = explanation['cam']
                    assert cam.shape == (224, 224) and np.isfinite(cam).all()
                    assert explanation['overlay'].shape == array.shape
                    with torch.inference_mode():
                        after = view(tensor).float().softmax(-1).cpu()
                    torch.testing.assert_close(reference, after, rtol=1e-5, atol=1e-6)
                    directory = output/record['id']/Path(name).stem
                    directory.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(explanation['overlay']).save(directory/f'{method}_{target}_overlay.png')
                    Image.fromarray(explanation['heatmap']).save(directory/f'{method}_{target}_heatmap.png')
                    np.save(directory/f'{method}_{target}_map.npy', cam)
                    detail = explanation['details']
                    row = {'model': record['id'], 'input': name, 'method': method, 'target': target,
                           'milliseconds': milliseconds, 'shape': list(cam.shape), 'finite': True,
                           'prediction_unchanged': True, 'details': detail,
                           'fp32_vs_deployed_fake_probability_difference': abs(float(reference[0, class_mapping['FAKE']])-result['fake_probability'])}
                    if target == result['prediction']:
                        row['deletion_check'] = check_attribution(view, tensor, class_mapping[target], cam)
                    rows.append(row)
                    save(output/'checks.json', rows)
            print(f"XAI {record['id']} {name}: {len(predictor.methods)} methods, both targets checked", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def video_checks(output, all_methods=False):
    output.mkdir(parents=True, exist_ok=False)
    with (RUN/'manifest.csv').open(newline='', encoding='utf-8') as stream:
        manifest = list(csv.DictReader(stream))
    selected = [sorted([row for row in manifest if row['split'] == 'validation' and row['label'] == label], key=lambda row: row['filename'])[0] for label in ['REAL', 'FAKE']]
    report = {'selection': 'First filename alphabetically per class in validation; not selected for correct predictions', 'videos': [], 'errors': []}
    for record in catalog('Video'):
        predictor = create_predictor(record)
        with (RUN/record['id']/'validation_predictions.csv').open(newline='', encoding='utf-8') as stream:
            predictions = {row['filename']: row for row in csv.DictReader(stream)}
        for row in selected:
            result, elapsed = timed(lambda: predictor.predict_video(row['path'], method=None))
            indices = [frame['frame_index'] for frame in result['sampled_frames']]
            assert indices == normalized_indices(result['metadata']['frame_count'])
            assert result['prepared_frames'].shape == (8, 224, 224, 3)
            saved_probability = float(predictions[row['filename']]['fake_probability'])
            assert abs(saved_probability-result['fake_probability']) < 2e-4, (record['id'], saved_probability, result['fake_probability'])
            position = 0
            method = 'gradcam' if 'gradcam' in predictor.methods else 'integrated_gradients'
            examples, xai_ms = timed(lambda: predictor.explain_result(result, method, frame_indices=[position]))
            view = predictor.attribution_model(torch.from_numpy(result['feature_context']).to(predictor.device), position)
            tensor = predictor.tensor(result['prepared_frames'][position:position+1])
            with torch.inference_mode():
                unscaled = view(tensor)
                explained_score = float(unscaled.softmax(-1)[0, 1])
            assert abs(explained_score-result['fake_probability']) < .003, (record['id'], explained_score, result['fake_probability'])
            Image.fromarray(examples[0]['overlay']).save(output/f"{record['id']}_{row['label']}_overlay.png")
            entry = {'model': record['id'], 'filename': row['filename'], 'expected': row['label'],
                     'prediction': result['prediction'], 'fake_probability': result['fake_probability'],
                     'saved_validation_probability': saved_probability, 'input_shape': [8, 3, 224, 224],
                     'indices': indices, 'crop_methods': [frame['crop_method'] for frame in result['sampled_frames']],
                     'milliseconds': elapsed, 'xai_ms': xai_ms, 'temperature': result['temperature'],
                     'timings': result['timings'], 'explanation_scope': examples[0]['coverage'],
                     'complete_selected_decision_probability': explained_score}
            report['videos'].append(entry)
            if all_methods and record['id'] in {'S8','S4','S0'}:
                entry['all_video_methods']=[]
                for extra_method in predictor.methods:
                    extra=predictor.explain_result(result,extra_method,frame_indices=[0])[0]
                    assert np.isfinite(extra['cam']).all()
                    entry['all_video_methods'].append({'method':extra_method,'details':extra['details']})
            print(f"Video {record['id']} {row['label']}: {result['prediction']} pFAKE={result['fake_probability']:.4f} in {elapsed:.0f} ms", flush=True)
            if record['id'] == 'S6' and row['label'] == 'REAL':
                repeated = predictor.predict_video(row['path'], method=None)
                assert repeated['fake_probability'] == result['fake_probability']
                assert [frame['frame_index'] for frame in repeated['sampled_frames']] == indices
                entry['deterministic_repeat'] = True
                all_methods = []
                for method in predictor.methods:
                    explanation = predictor.explain_result(result, method, 'REAL', frame_indices=[0])[0]
                    assert np.isfinite(explanation['cam']).all()
                    all_methods.append({'method': method, 'details': explanation['details']})
                entry['all_video_methods'] = all_methods
        del predictor
        torch.cuda.empty_cache()
        save(output/'checks.json', report)
    # Lossless container remux if FFmpeg is available; OpenCV otherwise encodes
    # a new QA clip whose result is not substituted into benchmark metrics.
    source = selected[0]['path']
    mov = output/'real_container_qa.mov'
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        subprocess.run([ffmpeg, '-v', 'error', '-i', source, '-c', 'copy', str(mov)], check=True)
        report['mov_mode'] = 'lossless remux'
    else:
        capture = cv2.VideoCapture(source)
        fps = capture.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(str(mov), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                 (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        assert writer.isOpened(), 'MOV writer unavailable'
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            capture.release(); writer.release()
        report['mov_mode'] = 'OpenCV mp4v re-encode; not pixel-equivalent benchmark input'
    predictor = create_predictor(next(record for record in catalog('Video') if record['id'] == 'S6'))
    mov_result = predictor.predict_video(mov, method=None)
    report['mov'] = {'decoded': True, 'prediction': mov_result['prediction'], 'fake_probability': mov_result['fake_probability'], 'metadata': mov_result['metadata']}
    corrupt = output/'corrupt.mp4'
    corrupt.write_bytes(b'not a video')
    for name, path in [('corrupt', corrupt), ('missing', output/'missing.mov')]:
        try:
            predictor.predict_video(path, method=None)
        except (ValueError, OSError, RuntimeError, cv2.error) as error:
            report['errors'].append({'input': name, 'handled': True, 'message': str(error)})
        else:
            raise AssertionError('Invalid video accepted')
    save(output/'checks.json', report)
    return report


def preprocessing_checks(output):
    output.mkdir(parents=True, exist_ok=False)
    records = {row['id']: row for row in catalog('Video')}
    baseline, spatial = create_predictor(records['S0']), create_predictor(records['S2'])
    rows = []
    for name in ['random-person.jpeg', 'real.jpeg']:
        image = Image.open(PICTURES/name).convert('RGB')
        raw = baseline.predict_image(image, method=None)['prepared_input']
        aligned = spatial.predict_image(image, method=None)['prepared_input']
        for predictor in [baseline, spatial]:
            for preprocessing, array in [('full_frame', raw), ('DFDC_landmark_alignment', aligned)]:
                features = predictor.feature_sequence(array[None])
                p, _, _ = predictor.score(features, image=True)
                rows.append({'model': predictor.id, 'input': name, 'preprocessing': preprocessing, 'fake_probability': p})
    save(output/'checks.json', rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['images', 'video', 'preprocessing'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--all-methods',action='store_true',help='Also exercise all methods on the legacy, ViT and combined video architectures')
    args = parser.parse_args()
    configure_runtime(); torch.set_num_threads(2); cv2.setNumThreads(2)
    function = {'images': image_checks, 'video': video_checks, 'preprocessing': preprocessing_checks}[args.phase]
    if args.phase=='video':
        function(args.output or ROOT/args.phase,all_methods=args.all_methods)
    else:
        function(args.output or ROOT/args.phase)
