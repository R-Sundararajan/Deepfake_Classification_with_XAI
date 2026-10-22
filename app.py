"""Streamlit demo: simple model names and automatic concurrent XAI galleries."""
import json
import time
from concurrent.futures import as_completed, CancelledError
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from config import PROJECT_ROOT
from inference.catalog import catalog, grouped, create_predictor, display_name, model_description, model_tag_legend
from inference.forensics import preserve_upload, image_metadata, video_container
from explainability.api import METHODS
from explainability.gradcam import create_visualizations
from explainability.guides import GUIDES, decision_description
from explainability.suite import ParallelExplainer
from tools.runtime import configure_runtime

configure_runtime()
st.set_page_config(page_title='Explainable Deepfake Classifier', page_icon='🔎', layout='wide')
CACHE_VERSION = 'automatic-gallery-v1'


def render_page_notes(notes):
    with st.expander('Notes and warnings', expanded=False):
        for severity, message in notes:
            (st.warning if severity == 'warning' else st.info)(message)
        st.warning('Scores express model uncertainty. No validated inconclusive threshold is deployed. Heatmaps and attention show model behavior; they do not prove manipulation or diagnose a facial artifact.')
        st.info('Heatmap colors are normalized separately for each method. Brightness and highlighted area cannot be compared as confidence scores. Sparse support is possible for both REAL and FAKE predictions; full-face coverage is not a criterion for authenticity.')


@st.cache_resource(show_spinner=False, max_entries=2)
def load_predictor(identifier, checkpoint_hash, source, checkpoint, version):
    return create_predictor({'id': identifier, 'source': source, 'checkpoint': checkpoint})


def predictor_for(record):
    return load_predictor(record['id'], record['checkpoint_sha256'], record['source'], record['checkpoint'], CACHE_VERSION)


@st.cache_resource(show_spinner=False)
def explanation_runner():
    return ParallelExplainer(workers=2, max_suites=4, ttl=1800)


@st.cache_data(show_spinner=False, max_entries=4, ttl=1800)
def classify(identifier, checkpoint_hash, source, checkpoint, path, modality, version):
    predictor = load_predictor(identifier, checkpoint_hash, source, checkpoint, version)
    with predictor.lock:
        if modality == 'Image':
            with Image.open(path) as image:
                return predictor.predict_image(image, method=None)
        return predictor.predict_video(path, method=None)


@st.cache_data(show_spinner=False, max_entries=8, ttl=1800)
def cached_explanation(identifier, checkpoint_hash, source, checkpoint, path, modality, target, method, parameters_json, positions, version):
    result = classify(identifier, checkpoint_hash, source, checkpoint, path, modality, version)
    predictor = load_predictor(identifier, checkpoint_hash, source, checkpoint, version)
    parameters = json.loads(parameters_json)
    with predictor.lock:
        if modality == 'Video':
            parameters['frame_indices'] = list(positions)
        return predictor.explain_result(result, method, target, **parameters)


def add_explanation_notes(explanation, notes):
    details = explanation.get('details', {})
    method = METHODS[explanation['method']][0]
    prefix = f"{method}" + (f" · frame {explanation['frame_index']}" if 'frame_index' in explanation else '')
    if details.get('constant_map') or details.get('low_signal'):
        notes.append(('warning', f'{prefix}: constant or negligible attribution; spatial localization is unavailable.'))
    if details.get('converged') is False:
        notes.append(('warning', f"Integrated Gradients did not meet its completeness tolerance at {details['steps']} steps. Treat the map as numerically inconclusive; inspect the convergence diagnostics."))
    if details.get('weighted_surrogate_r2', 1.) < .5:
        notes.append(('warning', f'{prefix}: weak local surrogate fit for this input. Its highlighted regions may be unreliable.'))


def show_explanation(explanation, strength, notes):
    heatmap, overlay = create_visualizations(explanation['original'], explanation['cam'], strength, signed=explanation.get('signed', False))
    columns = st.columns(3)
    columns[0].image(explanation['original'], caption='Actual model input', width='stretch')
    columns[1].image(heatmap, caption='Attribution map', width='stretch')
    columns[2].image(overlay, caption='Explanation overlay', width='stretch')
    st.caption(explanation['legend'])
    st.caption(explanation['coverage'])
    details = explanation['details']
    if details.get('constant_map') or details.get('low_signal'):
        st.caption('Localization unavailable · negligible or constant attribution. See Notes and warnings.')
    elif details.get('converged') is False:
        st.caption('Numerically inconclusive · completeness check failed. See Notes and warnings.')
    elif details.get('weighted_surrogate_r2', 1.) < .5:
        st.caption('Weak local fit · see Notes and warnings before interpreting this map.')
    with st.expander('Explanation diagnostics'):
        st.json(explanation['details'])
    add_explanation_notes(explanation, notes)


page_notes = []
st.title('Explainable Deepfake Classification')
st.caption('Image and sampled-video classification with visual model evidence')
kind = st.radio('Input type', ['Image', 'Video'], horizontal=True)
experimental = st.checkbox('Include unbenchmarked DFDC image transfers', value=False) if kind == 'Image' else False
records = catalog(kind, experimental)
if not records:
    st.error('No compatible, verified checkpoints are available.')
    render_page_notes(page_notes)
    st.stop()
families = grouped(records)
lookup = {record['id']: record for record in records}

def variant_label(identifier):
    record = lookup[identifier]
    display_metrics = record.get('full_validation', record['validation'])
    accuracy = display_metrics.get('accuracy') if record['ranked'] else None
    split_label = 'full validation accuracy' if record.get('full_validation') else 'validation accuracy'
    digits = 3 if record.get('full_validation') else 2
    score = f' · {accuracy:.{digits}%} {split_label}' if accuracy is not None else ' · accuracy not measured'
    suffix = ' (recommended)' if identifier == records[0]['id'] else ''
    if not record['ranked']:
        suffix += ' (experimental image transfer)'
    return display_name(record) + score + suffix

selected = st.selectbox('Detection model', list(lookup), format_func=variant_label)
st.caption(model_tag_legend(records))
record = lookup[selected]
st.caption(model_description(record))
if record['ranked']:
    metrics = record['validation']
    validation_label = 'Common 2,000-image validation' if record.get('full_validation') else 'Validation'
    st.caption(f"{validation_label}: balanced accuracy {metrics['balanced_accuracy']:.2%} · accuracy {metrics['accuracy']:.2%} · macro-F1 {metrics['macro_f1']:.3f} · {record['evaluation_unit']} level")
    if record.get('full_validation'):
        full, held_out = record['full_validation'], record['held_out_test']
        st.caption(f"Full validation ({full['samples']:,} images): accuracy {full['accuracy']:.3%}, balanced accuracy {full['balanced_accuracy']:.3%}. Held-out test ({held_out['samples']:,} images): accuracy {held_out['accuracy']:.3%}, balanced accuracy {held_out['balanced_accuracy']:.3%}.")
else:
    page_notes.append(('info', 'The reported DFDC validation metrics are video-level. Image transfer of this DFDC-trained spatial head has not been benchmarked.'))
if kind == 'Image' and selected in {'S6', 'S7', 'S8'}:
    spatial_id = 'S5' if selected == 'S8' else 'S2'
    page_notes.append(('info', f'Image mode uses the {spatial_id} spatial checkpoint. The temporal branch is bypassed; this fallback is separate from the video benchmark.'))
page_notes.append(('info', 'Model families are ordered by their best compatible validation balanced accuracy, with macro-F1 and ROC-AUC breaking ties. Family means include only completed runs on the same validation protocol. Image and video rankings are separate.'))
with st.expander('Model and evaluation details'):
    st.write(record['training_dataset'])
    st.write(record['evaluation_dataset'])
    st.json({'checkpoint_id': selected, 'checkpoint_sha256': record['checkpoint_sha256'], 'checkpoint_path': record['checkpoint'],
             'architecture': record['config'], 'validation': record['validation'], 'calibration': record['calibration'],
             'protocol': record['evaluation_protocol'], 'limitations': record.get('limitations', 'Small local subset; source grouping does not establish actor disjointness. Image transfer is a separate domain.')})
    if record.get('full_validation'):
        st.json({'full_validation': record['full_validation'], 'held_out_test': record['held_out_test']})
    st.dataframe([{'Family': group['family'], 'Best model': display_name(group['best']),
                   'Best validation BA': group['best']['validation'].get('balanced_accuracy'),
                   'Mean validation BA': group['mean_balanced_accuracy'], 'Comparable completed runs': group['comparable_runs']}
                  for group in families], hide_index=True, width='stretch')
    if kind == 'Image':
        page_notes.append(('info', 'Image validation uses the same 2,000-image subset with each model’s native preprocessing. The preserved ResNet used 100,000 training images; E01–E04 used 12,000. DFDC transfer variants trained on videos. Their training budgets and domains differ.'))

uploaded = st.file_uploader('Upload a face image' if kind == 'Image' else 'Upload a video', type=['jpg', 'jpeg', 'png'] if kind == 'Image' else ['mp4', 'mov'])
if uploaded is None:
    render_page_notes(page_notes)
    st.stop()
provenance = None
try:
    path, provenance = preserve_upload(uploaded.getvalue(), uploaded.name, kind)
    provenance.update(image_metadata(path) if kind == 'Image' else video_container(path))
    tick = time.perf_counter()
    with st.spinner('Classifying...'):
        result = classify(selected, record['checkpoint_sha256'], record['source'], record['checkpoint'], str(path), kind, CACHE_VERSION)
    retrieval_ms = (time.perf_counter()-tick)*1000
except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, RuntimeError, cv2.error) as error:
    st.error(f'Input could not be classified: {error}')
    if provenance:
        with st.expander('Input provenance and decode error', expanded=True):
            st.json({**provenance, 'decode_error': str(error)})
    render_page_notes(page_notes)
    st.stop()
columns = st.columns(3)
columns[0].metric('Prediction', result['prediction'])
columns[1].metric('FAKE probability', f"{result['fake_probability']:.2%}")
columns[2].metric('REAL probability', f"{result['real_probability']:.2%}")
page_notes.append(('info', result['confidence_label']))
with st.expander('Forensic input information'):
    st.json(provenance)
    st.download_button('Download preserved original', uploaded.getvalue(), file_name=Path(uploaded.name).name)
with st.expander('Inference timings'):
    st.json({'prediction_seconds': result['elapsed_seconds'], 'current_cached_lookup_or_compute_ms': retrieval_ms, **result.get('timings', {})})
if kind == 'Image':
    columns = st.columns(2)
    columns[0].image(result['original_input'], caption='Original upload', width='stretch')
    columns[1].image(result['prepared_input'], caption='Actual model input crop', width='stretch')
    with st.expander('Preprocessing details'):
        st.write(result['crop_method'])
        st.json(result.get('crop_metadata', {}))
    if result['crop_method'] == 'no_face_full_frame_fallback':
        page_notes.append(('warning', 'No face detected. The full image was used as an explicit fallback.'))
    positions = ()
else:
    provenance.update(result['metadata'])
    with st.expander('Video playback'):
        st.video(str(path))
    st.write(result['aggregation'])
    st.subheader('How the sampled frames scored')
    st.write('Why this plot: shows how appearance scores vary across the eight sampled moments. What it depicts: each point is the spatial branch’s FAKE probability at that time; the horizontal position is time in seconds. The final video prediction can also use motion.')
    frame_rows = [{key: value for key, value in row.items() if key != 'crop_metadata'} for row in result['sampled_frames']]
    st.dataframe(frame_rows, hide_index=True, width='stretch')
    st.line_chart(pd.DataFrame(frame_rows).set_index('seconds')[['fake_probability']], y_label='Spatial FAKE score')
    page_notes.extend([('info', result['timeline_caption']), ('info', result['limitations'])])
    with st.expander('Spatial / temporal decision and sensitivity'):
        contributions = result['contributions']
        st.json(contributions)
        if 'position_mean_replacement_fake_probability_drops' in contributions:
            st.write('Why this plot: tests sensitivity to each sampled position. What it depicts: the change in the complete system’s FAKE score when that position’s motion features are replaced by the training mean. Positive bars mean replacement lowered the score; negative bars mean it raised it. These scores are before temperature calibration.')
            st.bar_chart(pd.DataFrame({'seconds': [row['seconds'] for row in frame_rows], 'FAKE score drop before calibration': contributions['position_mean_replacement_fake_probability_drops']}).set_index('seconds'))
        page_notes.append(('info', contributions.get('limitation', 'Frame scores are spatial predictions; representative-frame attribution differentiates the selected complete video decision.')))
    if any(row['crop_method'] == 'no_face_full_frame_fallback' for row in frame_rows):
        page_notes.append(('warning', 'Face detection failed on some sampled frames. Those frames used the full image; inspect each crop status.'))
    representative = max(range(len(frame_rows)), key=lambda index: frame_rows[index]['fake_probability'] if result['prediction'] == 'FAKE' else 1-frame_rows[index]['fake_probability'])
    with st.expander('Video explanation frames (optional)'):
        st.caption('The default is the sampled frame whose appearance score most supports the video prediction. Every method explains the complete video decision while the other sampled embeddings stay fixed.')
        positions = tuple(st.multiselect('Frames to explain (up to 3)', list(range(len(frame_rows))), default=[representative], max_selections=3,
                                        format_func=lambda index: f"Frame {frame_rows[index]['frame_index']} at {frame_rows[index]['seconds']:.2f} s"))
    if positions:
        st.image([result['prepared_frames'][position] for position in positions], caption=[f"Sample at {frame_rows[position]['seconds']:.2f} s" for position in positions], width=224)

st.subheader(f"Why the model predicted {result['prediction']}")
st.write(decision_description(result['prediction']))
st.caption('All compatible explanations run automatically. Scroll to compare the views. The class being explained is chosen from the prediction; there is no extra REAL/FAKE setting.')
with st.expander('Display settings'):
    strength = st.slider('Overlay opacity', .1, .9, .45, .05)
    st.caption('Opacity changes reuse the completed maps.')
if kind == 'Video' and not positions:
    st.caption('Choose at least one sampled frame in Video explanation frames to show its maps.')
else:
    suite = explanation_runner().start(record, result, str(path), kind, positions, CACHE_VERSION, predictor=predictor_for(record))
    progress = st.progress(0, text=f'Preparing {len(suite.futures)} explanation views…')
    slots = {}
    for method in suite.futures:
        guide = GUIDES[method]
        with st.container(border=True):
            st.subheader(guide['title'])
            st.write('**Why this plot:** ' + guide['why'])
            st.write('**What it depicts:** ' + guide['depicts'])
            if method != 'attention_rollout':
                st.caption(f"Explaining the model’s {suite.target} decision.")
            slots[method] = st.empty()
            slots[method].caption('Preparing this view…')
    completed = 0
    future_methods = {future: method for method, future in suite.futures.items()}
    for future in as_completed(future_methods):
        method = future_methods[future]
        with slots[method].container():
            try:
                explanations = future.result()
                if kind == 'Image':
                    show_explanation(explanations, strength, page_notes)
                else:
                    for explanation in explanations:
                        st.markdown(f"**Frame {explanation['frame_index']} at {explanation['seconds']:.2f} s**")
                        show_explanation(explanation, strength, page_notes)
            except (OSError, ValueError, RuntimeError, cv2.error, CancelledError) as error:
                st.caption('This view is unavailable. See Notes and warnings for details.')
                page_notes.append(('warning', f'{METHODS[method][0]} could not be generated: {error}'))
        completed += 1
        progress.progress(completed / len(suite.futures), text=f'{completed} of {len(suite.futures)} explanation views ready')
    progress.empty()
render_page_notes(page_notes)
