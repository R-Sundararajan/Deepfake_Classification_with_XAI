"""Exercise automatic image/video XAI, cached display changes and plain names."""
import csv
import json
from pathlib import Path
import threading
import time
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from config import PROJECT_ROOT
from explainability.suite import ParallelExplainer
from inference.media import MediaPredictor
from inference.unified import ResearchPredictor
from tools.runtime import configure_runtime
from training.video_protocol import RUN


def run(output):
    configure_runtime()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    checks, calls, suites = [], [], []
    counters = {'predictions': 0, 'methods': 0}
    original_image = ResearchPredictor.predict_image
    original_video = ResearchPredictor.predict_video
    original_media = MediaPredictor.predict_image
    original_start = ParallelExplainer.start
    lock = threading.Lock()

    def count_prediction(function):
        def wrapped(self, *args, **kwargs):
            with lock:
                counters['predictions'] += 1
            return function(self, *args, **kwargs)
        return wrapped

    def count_explanation(function):
        def wrapped(self, image, target, method, *args, **kwargs):
            start = time.perf_counter()
            result = function(self, image, target, method, *args, **kwargs)
            assert np.isfinite(result['cam']).all()
            assert result['cam'].shape == (224, 224)
            with lock:
                counters['methods'] += 1
                calls.append({'method': method, 'target': result['details']['target_class'],
                              'worker': threading.current_thread().name, 'started': start,
                              'finished': time.perf_counter(), 'details': result['details']})
            return result
        return wrapped

    def capture_suite(self, *args, **kwargs):
        suite = original_start(self, *args, **kwargs)
        suites.append(suite)
        return suite

    patches = [patch.object(ResearchPredictor, 'predict_image', count_prediction(original_image)),
               patch.object(ResearchPredictor, 'predict_video', count_prediction(original_video)),
               patch.object(MediaPredictor, 'predict_image', count_prediction(original_media)),
               patch.object(ResearchPredictor, 'explain', count_explanation(ResearchPredictor.explain)),
               patch.object(MediaPredictor, 'explain', count_explanation(MediaPredictor.explain)),
               patch.object(ParallelExplainer, 'start', capture_suite)]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        app = AppTest.from_file(str(PROJECT_ROOT / 'app.py'), default_timeout=180)

        def check(name, action=None, methods=None):
            before = dict(counters)
            tick = time.perf_counter()
            if action:
                action()
            app.run()
            assert not app.exception, [item.message for item in app.exception]
            footer = list(app.main.children.values())[-1]
            assert footer.label == 'Notes and warnings' and not footer.proto.expanded
            assert len(app.warning) == len(footer.warning)
            assert all(widget.label not in {'Explanation method', 'Trained variant', 'Model family'} for widget in app.selectbox)
            assert all(widget.label != 'Explain output' for widget in app.radio)
            assert all(widget.label != 'Generate explanation' for widget in app.button)
            if methods:
                suite = suites[-1]
                assert len(suite.futures) == methods
                for future in suite.futures.values():
                    explanation = future.result()
                    for item in explanation if isinstance(explanation, list) else [explanation]:
                        assert item['details']['target_class'] == app.metric[0].value
                headings = [item.value for item in app.subheader]
                assert sum(' · ' in heading for heading in headings) == methods, headings
            row = {'case': name, 'milliseconds': (time.perf_counter() - tick) * 1000,
                   'new_predictions': counters['predictions'] - before['predictions'],
                   'new_maps': counters['methods'] - before['methods']}
            checks.append(row)
            print(json.dumps(row), flush=True)
            return footer

        def model():
            return next(item for item in app.selectbox if item.label == 'Detection model')

        def upload(path, mime):
            path = Path(path)
            app.file_uploader[0].set_value((path.name, path.read_bytes(), mime))

        check('initial')
        assert model().value == 'S0'
        assert all(not label.startswith(('S0', 'S1', 'S2', 'E0')) for label in model().options)
        pictures = Path(r'C:\Users\Rangarajan\OneDrive\Pictures\Saved Pictures')
        check('synthetic_auto_six', lambda: upload(pictures / 'random-person.jpeg', 'image/jpeg'), 6)
        assert app.metric[0].value == 'FAKE'
        first_prediction = [item.value for item in app.metric]
        assert checks[-1]['new_predictions'] == 1 and checks[-1]['new_maps'] == 6
        check('rerender_cached', methods=6)
        assert checks[-1]['new_predictions'] == checks[-1]['new_maps'] == 0
        check('opacity_cached', lambda: app.slider[0].set_value(.7), 6)
        assert checks[-1]['new_predictions'] == checks[-1]['new_maps'] == 0
        assert [item.value for item in app.metric] == first_prediction
        check('real_auto_six', lambda: upload(pictures / 'real.jpeg', 'image/jpeg'), 6)
        assert app.metric[0].value == 'REAL'
        check('model_change_automatic', lambda: model().set_value('E02_efficientnet_b0'), 6)
        check('experimental_options', lambda: app.checkbox[0].set_value(True))
        footer = check('hybrid_image_eight', lambda: model().set_value('S8'), 8)
        assert any('S5 spatial checkpoint' in item.value for item in footer.info)
        check('video_initial', lambda: app.radio[0].set_value('Video'))
        assert model().value == 'S6'
        with (RUN / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
            manifest = list(csv.DictReader(stream))
        clips = {label: sorted([row for row in manifest if row['split'] == 'validation' and row['label'] == label], key=lambda row: row['filename'])[0] for label in ['REAL', 'FAKE']}
        check('video_real_auto_six', lambda: upload(clips['REAL']['path'], 'video/mp4'), 6)
        assert app.metric[0].value == 'REAL'
        check('video_cache', methods=6)
        assert checks[-1]['new_predictions'] == checks[-1]['new_maps'] == 0
        positions = app.multiselect[0].value
        extra = next(position for position in range(8) if position not in positions)
        check('video_extra_frame_automatic', lambda: app.multiselect[0].set_value(positions + [extra]), 6)
        assert checks[-1]['new_predictions'] == 0 and checks[-1]['new_maps'] == 12
        check('video_fake_auto_six', lambda: upload(clips['FAKE']['path'], 'video/mp4'), 6)
        assert app.metric[0].value == 'FAKE'
        check('video_hybrid_eight', lambda: model().set_value('S8'), 8)
        check('video_transformer_five', lambda: model().set_value('S4'), 5)
        check('corrupt_video', lambda: app.file_uploader[0].set_value(('corrupt.mp4', b'not a video', 'video/mp4')))
        assert len(app.error) == 1

    overlaps = [(a['method'], b['method']) for a in calls for b in calls
                if a['worker'] != b['worker'] and a['started'] < b['finished'] and b['started'] < a['finished']]
    assert overlaps, 'No concurrent attribution calls observed'
    report = {'checks': checks, 'calls': calls, 'counters': counters,
              'concurrent_pairs_observed': len(overlaps) // 2,
              'image_prediction': first_prediction,
              'timing_scope': 'AppTest wall time, includes model load, GPU warmup and rendering/serialization; not browser paint.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{len(checks)} UI cases passed; {len(calls)} maps checked.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/demo_presentation_20261020/app_qa.json')
    run(parser.parse_args().output)
