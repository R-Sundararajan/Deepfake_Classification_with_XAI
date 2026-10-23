"""Verify the fixed candidate through the real demo path before activation."""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from config import PROJECT_ROOT
from evaluation.dfdc import binary_metrics
import inference.catalog as registry
from explainability.suite import ParallelExplainer
from tools.runtime import configure_runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def repair_precision(value):
    # The first evaluator revision collided with the fake-class precision key.
    # Recover it exactly from the recorded confusion matrix, without new inference.
    if isinstance(value, dict):
        if value.get('precision') in ('FP32', 'AMP FP16') and 'confusion_matrix_real_fake' in value:
            value['inference_precision'] = value['precision']
            matrix = value['confusion_matrix_real_fake']; fp, tp = matrix[0][1], matrix[1][1]
            value['precision'] = tp/(tp+fp) if tp+fp else 0.
        for item in value.values():
            repair_precision(item)
    elif isinstance(value, list):
        for item in value:
            repair_precision(item)


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def run(root):
    configure_runtime()
    root = root.resolve()
    assert root.is_relative_to(PROJECT_ROOT/'results')
    write(root/'processes.json',[{'pid':os.getpid(),'role':'verification','started_utc':datetime.now(timezone.utc).isoformat()}])
    for path in root.glob('*.json'):
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        repair_precision(value)
        if path.name in {'baseline_validation_fp32.json', 'selection.json', 'history.json'} or path.name.startswith('epoch_'):
            write(path, value)
    status = json.loads((root/'final_status.json').read_text(encoding='utf-8'))
    assert status['completed'] and status['improved'] and status['test_evaluated']
    checkpoint = Path(status['checkpoint']); checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash == status['checkpoint_sha256']
    baseline = json.loads((root/'baseline_validation_fp32.json').read_text(encoding='utf-8'))
    full_validation, test = status['validation'], status['test']
    assert full_validation['samples'] == test['samples'] == baseline['samples'] == 20000
    assert full_validation['split'] == 'validation' and test['split'] == 'test'
    assert full_validation['balanced_accuracy'] > baseline['balanced_accuracy']
    for metrics in [full_validation, test, baseline]:
        assert isinstance(metrics['precision'], (int, float))
    full_rows = read_rows(root/'candidate_validation_fp32.csv')
    reference_rows = read_rows(PROJECT_ROOT/'results/execution_20261019/E00_validation/validation_predictions.csv')
    lookup = {str(Path(row['input']).resolve()).casefold(): row for row in full_rows}
    selected = [lookup[str(Path(row.get('path', row.get('input'))).resolve()).casefold()] for row in reference_rows]
    assert len(selected) == 2000
    assert all(a['label'] == b['label'] for a,b in zip(selected, reference_rows))
    common = binary_metrics([int(row['label']=='FAKE') for row in selected], [float(row['fake_probability']) for row in selected])
    common.update(samples=2000, completed=True, checkpoint_sha256=checkpoint_hash, split='validation')
    reference = json.loads((PROJECT_ROOT/'results/execution_20261019/E00_validation/summary.json').read_text(encoding='utf-8'))
    assert common['balanced_accuracy'] > reference['validation']['balanced_accuracy']
    record = {
        'id': 'R18_FT_20261021', 'source': 'image_continuation', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash, 'config': {'model':'ResNet-18 continued fine-tuning', 'backbones':['resnet18'],
        'preprocess':'full frame', 'model_spec':{'backbone':'resnet18','attention':'none'},
        'augmentation':'original flip/color/rotation', 'continued_finetuning':True},
        'validation':common, 'full_validation':full_validation, 'held_out_test':test,
        'calibration':{'temperature':1.,'scope':'uncalibrated images; fixed threshold 0.5'},
        'ranked':True, 'evaluation_unit':'image', 'evaluation_protocol':'Real vs Fake Faces validation; '+reference['manifest_sha256'],
        'evaluation_dataset':'Common 2,000-image comparison cohort, plus complete 20,000-image validation and final 20,000-image test',
        'training_dataset':'Original 100,000-image training split; continued from preserved ResNet-18',
        'training_image_count':100000, 'family':'ResNet / CNN',
        'limitations':'Same dataset domain as the image baseline. Classification heatmaps are not verified edit masks.'}
    gates={'full_validation_improved':True,'common_validation_improved':True,'reload':False,'inference':False,'xai':False,'ui':False,'latency':False}
    predictor = registry.create_predictor(record)
    assert predictor.checkpoint_sha256 == checkpoint_hash
    gates['reload'] = True
    pictures = Path(r'C:\Users\Rangarajan\OneDrive\Pictures\Saved Pictures')
    sanity = []
    runner = ParallelExplainer()
    try:
        for name in ['random-person.jpeg', 'real.jpeg']:
            path = pictures/name
            with Image.open(path) as image:
                tick=time.perf_counter(); result=predictor.predict_image(image,method=None); elapsed=time.perf_counter()-tick
            suite=runner.start(record,result,str(path),'Image',predictor=predictor)
            maps={method:future.result(timeout=120) for method,future in suite.futures.items()}
            assert len(maps)==6 and all(np.isfinite(item['cam']).all() and item['cam'].shape==(224,224) for item in maps.values())
            nonconstant=sum(np.std(item['cam'])>1e-8 for item in maps.values())
            assert nonconstant>=3, 'Insufficient usable localization views'
            with Image.open(path) as image:
                repeat_tick=time.perf_counter(); repeated=predictor.predict_image(image,method=None); warm_seconds=time.perf_counter()-repeat_tick
            assert abs(repeated['fake_probability']-result['fake_probability'])<1e-7
            assert warm_seconds<1., f'Warm prediction exceeds one second: {warm_seconds}'
            sanity.append({'input':name,'prediction':result['prediction'],'fake_probability':result['fake_probability'],
                           'prediction_seconds':elapsed,'warm_prediction_seconds':warm_seconds,'nonconstant_methods':int(nonconstant),
                           'methods':{method:item['details'] for method,item in maps.items()}})
            print(f'Candidate {name}: {result["prediction"]}; {nonconstant}/6 nonconstant maps.',flush=True)
    finally:
        runner.close()
    gates['inference']=gates['xai']=True
    original_catalog=registry.catalog
    def preview_catalog(modality='Image',include_experimental=False):
        rows=original_catalog(modality,include_experimental)
        return [record]+[row for row in rows if row['id']!=record['id']] if modality=='Image' else rows
    with patch.object(registry,'catalog',preview_catalog):
        app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=180).run()
        assert not app.exception,[item.message for item in app.exception]
        assert app.selectbox[0].value==record['id']
        for row in sanity:
            path=pictures/row['input']
            ui_tick=time.perf_counter()
            app.file_uploader[0].set_value((path.name,path.read_bytes(),'image/jpeg')).run()
            row['complete_gallery_seconds']=time.perf_counter()-ui_tick
            assert row['complete_gallery_seconds']<45., 'Automatic gallery too slow for demo'
            assert not app.exception,[item.message for item in app.exception]
            assert app.metric[0].value==row['prediction']
            assert sum(' · ' in heading.value for heading in app.subheader)==6
            footer=list(app.main.children.values())[-1]
            assert footer.label=='Notes and warnings' and not footer.proto.expanded
        gates['ui']=True
        gates['latency']=True
    write(root/'deployment.json',{'format':'verified_image_continuation_v1','verified':all(gates.values()),
                                 'record':record,'gates':gates,'baseline_full_validation':baseline,'sanity':sanity})
    print(json.dumps({'gates':gates,'full_validation':full_validation['accuracy'],'test':test['accuracy'],
                      'common_validation':common['accuracy']}),flush=True)


def verify_current_models(root, promote=False):
    """Verify and atomically rebind the two stable current-image aliases."""
    import gc
    import faulthandler
    import torch
    from config import VALID_PATH
    from training.demo_accuracy import IndexedFolder, write as atomic_write, sha
    from training.data import evaluation_transform
    from training.image_continuation import IDENTITIES, historical_hashes
    from inference.media import MediaPredictor
    configure_runtime()
    faulthandler.dump_traceback_later(60, repeat=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preflight = json.loads((root/'preflight.json').read_text(encoding='utf-8'))
    if not (root/'evaluation_complete.json').is_file():
        raise ValueError('Final evaluation must complete before deployment verification')
    if historical_hashes() != preflight['historical_hashes']:
        raise ValueError('Historical image/video evidence changed')
    if any(digest(source['checkpoint']) != source['sha256'] for source in preflight['sources'].values()):
        raise ValueError('A preserved initialization checkpoint changed')
    reference = json.loads((PROJECT_ROOT/'results/execution_20261019/E00_validation/summary.json').read_text(encoding='utf-8'))
    data = IndexedFolder(VALID_PATH, transform=evaluation_transform())
    records, sanity = [], []
    original_ids = [record['id'] for record in registry.catalog('Image')]
    gates = {'historical_evidence_unchanged': True, 'reload': False, 'preprocessing_and_labels': False,
             'xai_finite_aligned': False, 'xai_preserves_weights_predictions': False,
             'unique_existing_ids': False, 'streamlit_uploads': False, 'video_view': False}
    for identity in IDENTITIES:
        folder = root/identity
        status = json.loads((folder/'final_status.json').read_text(encoding='utf-8'))
        training = json.loads((folder/'training_finished.json').read_text(encoding='utf-8'))
        checkpoint = Path(status['checkpoint'])
        if sha(checkpoint) != status['checkpoint_sha256'] or not status['promotion_eligible']:
            raise ValueError('Checkpoint is ineligible or corrupt')
        backbone = 'resnet18' if identity == IDENTITIES[0] else 'efficientnet_b0'
        record = {
            'id': identity, 'source': 'image_continuation', 'checkpoint': str(checkpoint),
            'checkpoint_sha256': status['checkpoint_sha256'],
            'config': {'model': 'ResNet-18 continued fine-tuning' if backbone == 'resnet18' else 'EfficientNet-B0 full-dataset fine-tuning',
                       'backbones': [backbone], 'model_spec': {'backbone': backbone, 'attention': 'none', 'fusion': 'none'},
                       'preprocess': 'full frame', 'augmentation': 'original flip/color/rotation',
                       'continued_finetuning': training['selected_epoch'] > 0},
            'validation': status['common_validation'], 'full_validation': status['validation'], 'held_out_test': status['test'],
            'calibration': {'temperature': 1., 'scope': 'uncalibrated image probabilities; fixed threshold 0.5'},
            'ranked': True, 'evaluation_unit': 'image',
            'evaluation_protocol': 'Real vs Fake Faces validation; '+reference['manifest_sha256'],
            'evaluation_dataset': 'Common 2,000-image ranking cohort; separately labeled full 20,000 validation and 20,000 repeated test images',
            'training_image_count': 100000 if training['selected_epoch'] > 0 else training['warm_start']['previous_training_images'],
            'training_dataset': 'Real vs Fake Faces predefined 100,000-image training split; existing image checkpoint warm start',
            'training_summary': {**{k: v for k, v in training.items() if k not in {'history', 'selected_validation'}},
                                 'checkpoint_bytes': status['checkpoint_bytes']},
            'inference_benchmark': json.loads((folder/'inference_benchmark.json').read_text(encoding='utf-8')),
            'limitations': 'Within-dataset results. Test split was used historically; this repeat is not a newly independent sealed benchmark. Heatmaps show model influence, not verified edit masks.'}
        records.append(record)
        predictor = MediaPredictor(checkpoint)
        print(f'VERIFY {identity}: checkpoint reload passed', flush=True)
        if predictor.checkpoint_sha256 != record['checkpoint_sha256']:
            raise ValueError('Predictor loaded a different checkpoint')
        state = {k: v.detach().cpu().clone() for k, v in predictor.model.state_dict().items()}
        saved_rows = read_rows(folder/'candidate_validation_fp32.csv')
        saved_probabilities = {str(Path(r['input']).resolve()).casefold(): float(r['fake_probability']) for r in saved_rows}
        for index in [0, 10000]:
            image_path = Path(data.samples[index][0])
            with Image.open(image_path) as original:
                image = original.convert('RGB')
            result = predictor.predict_image(image, method=None)
            tensor = evaluation_transform()(image).unsqueeze(0).to(predictor.device)
            with torch.inference_mode():
                expected = float(predictor.model(tensor).softmax(1)[0, 0])
            if abs(expected-result['fake_probability']) > 2e-6 or abs(result['fake_probability']+result['real_probability']-1) > 1e-12:
                raise ValueError('UI and evaluation preprocessing/probabilities differ')
            if abs(saved_probabilities[str(image_path.resolve()).casefold()]-result['fake_probability']) > 2e-5:
                raise ValueError('Uploaded image does not reproduce saved FP32 prediction evidence')
            details = {}
            for method in predictor.methods:
                explanation = predictor.explain_result(result, method, steps=32, max_steps=128, samples=256, patch=32)
                cam = explanation['cam']
                if cam.shape != (224, 224) or not np.isfinite(cam).all():
                    raise ValueError('Invalid attribution: '+method)
                details[method] = explanation['details']
            repeated = predictor.predict_image(image, method=None)
            if abs(repeated['fake_probability']-result['fake_probability']) > 1e-7:
                raise ValueError('XAI changed predictions')
            sanity.append({'identity': identity, 'input': str(image_path), 'actual_label': 'FAKE' if index == 0 else 'REAL',
                           'prediction': result['prediction'], 'fake_probability': result['fake_probability'], 'methods': details})
        if any(not torch.equal(v.detach().cpu(), state[k]) for k, v in predictor.model.state_dict().items()):
            raise ValueError('XAI changed model parameters/buffers')
        print(f'VERIFY {identity}: predictions and six XAI methods passed on REAL/FAKE inputs; weights unchanged', flush=True)
        del predictor, tensor, state, explanation
        gc.collect(); torch.cuda.empty_cache()
    gates.update(reload=True, preprocessing_and_labels=True, xai_finite_aligned=True, xai_preserves_weights_predictions=True)
    # Preview via the same resolver used by catalog(), before activation.
    with patch.object(registry, 'current_image_candidates', return_value=records):
        catalog = registry.catalog('Image')
        ids = [r['id'] for r in catalog]
        if (len(ids) != len(set(ids)) or any(ids.count(i) != 1 for i in IDENTITIES)
                or set(ids) != set(original_ids)):
            raise ValueError('Duplicate or missing active identity')
        gates['unique_existing_ids'] = True
        print('VERIFY UI: starting empty image view', flush=True)
        app = AppTest.from_file(str(PROJECT_ROOT/'app.py'), default_timeout=120).run()
        print('VERIFY UI: empty image view returned', flush=True)
        if app.exception:
            raise AssertionError([e.message for e in app.exception])
        for record in records:
            print('VERIFY UI: selecting '+record['id'], flush=True)
            app.selectbox[0].set_value(record['id']).run()
            if app.exception:
                raise AssertionError([e.message for e in app.exception])
            path = Path(data.samples[0][0])
            print('VERIFY UI: uploading image for '+record['id'], flush=True)
            app.file_uploader[0].set_value((path.name, path.read_bytes(), 'image/jpeg')).run()
            if app.exception:
                raise AssertionError([e.message for e in app.exception])
            if sum(' · ' in heading.value for heading in app.subheader) != 6:
                raise AssertionError('Automatic six-method gallery did not render')
            options = list(app.selectbox[0].options)
            if len(options) != len(catalog):
                raise AssertionError('Selector option count differs from catalog')
            if not any(f"{record['full_validation']['accuracy']:.3%}" in str(option) for option in options):
                raise AssertionError('Updated full-validation score missing from selector')
            print(f'VERIFY {record["id"]}: actual Streamlit image upload and six-method gallery passed', flush=True)
        gates['streamlit_uploads'] = True
        video_app = AppTest.from_file(str(PROJECT_ROOT/'app.py'), default_timeout=240).run()
        video_app.radio[0].set_value('Video').run()
        if video_app.exception:
            raise AssertionError([e.message for e in video_app.exception])
        gates['video_view'] = True
    deployment = {'format': 'verified_current_image_models_v1', 'verified': all(gates.values()),
                  'gates': gates, 'records': records, 'verified_utc': datetime.now(timezone.utc).isoformat()}
    atomic_write(root/'deployment.json', deployment)
    verification = {'gates': gates, 'sanity': sanity, 'promotion_requested': promote, 'promoted': False}
    atomic_write(root/'verification.json', verification)
    if promote:
        # A single atomic pointer switches both aliases together. The legacy
        # ResNet pointer is refreshed as a backward-compatible rollback fallback.
        pointer = PROJECT_ROOT/'configs/runtime/current_image_models.json'
        legacy_pointer = PROJECT_ROOT/'configs/runtime/active_image_model.json'
        previous = {path: path.read_text(encoding='utf-8') if path.exists() else None for path in [pointer, legacy_pointer]}
        deployments = root/'deployments'
        deployments.mkdir(exist_ok=True)
        catalog_digest = sha(root/'deployment.json')
        immutable_deployment = deployments/f'{catalog_digest}.json'
        if not immutable_deployment.exists():
            atomic_write(immutable_deployment, deployment)
        if sha(immutable_deployment) != catalog_digest:
            raise ValueError('Immutable deployment summary differs')
        resnet_deployment = root/'resnet_deployment.json'
        atomic_write(resnet_deployment, {'format': 'verified_image_continuation_v1', 'verified': True,
                                        'gates': gates, 'record': records[0]})
        resnet_digest = sha(resnet_deployment)
        immutable_resnet = deployments/f'{resnet_digest}.json'
        if not immutable_resnet.exists():
            atomic_write(immutable_resnet, json.loads(resnet_deployment.read_text(encoding='utf-8')))
        try:
            atomic_write(pointer, {'format': 'current_image_models_pointer_v1', 'summary': str(immutable_deployment),
                                   'summary_sha256': catalog_digest})
            atomic_write(legacy_pointer, {'format': 'demo_image_pointer_v1', 'summary': str(immutable_resnet),
                                          'summary_sha256': resnet_digest})
            resolved = registry.catalog('Image')
            for record in records:
                actual = [r for r in resolved if r['id'] == record['id']]
                if len(actual) != 1 or actual[0]['checkpoint_sha256'] != record['checkpoint_sha256']:
                    raise ValueError('Promoted catalog does not resolve expected checkpoints')
        except Exception:
            for path, content in previous.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    temporary = path.with_suffix('.json.tmp')
                    temporary.write_text(content, encoding='utf-8'); temporary.replace(path)
            raise
        atomic_write(root/'promotion.json', {'completed': True, 'utc': datetime.now(timezone.utc).isoformat(),
                                            'pointer': str(pointer), 'identities': list(IDENTITIES),
                                            'deployment_sha256': catalog_digest})
        verification['promoted'] = True
        atomic_write(root/'verification.json', verification)
    print(json.dumps({'gates': gates, 'promoted': promote}), flush=True)
    faulthandler.cancel_dump_traceback_later()


def check_current_consistency(root):
    """Check catalog/UI inputs, tables, figures and saved predictions agree."""
    from training.image_continuation import IDENTITIES, historical_hashes, better
    from training.demo_accuracy import sha, write as atomic_write
    from inference.demo_candidate import current_image_candidates
    def load(path):
        return json.loads(path.read_text(encoding='utf-8'))
    preflight = load(root/'preflight.json')
    metrics = load(root/'current_metrics.json')
    figure_data = load(PROJECT_ROOT/'docs/figures/current_image_models/metrics.json')
    rows = read_rows(root/'current_metrics.csv')
    aliases = current_image_candidates()
    if {r['id'] for r in aliases} != set(IDENTITIES) or len(aliases) != 2:
        raise ValueError('Current pointer does not expose exactly two stable aliases')
    active = registry.catalog('Image')
    if len({r['id'] for r in active}) != len(active):
        raise ValueError('Catalog contains duplicate active identities')
    readme = (PROJECT_ROOT/'README.md').read_text(encoding='utf-8')
    for record in aliases:
        identity = record['id']; folder = root/identity
        status = load(folder/'final_status.json')
        plotted = figure_data['models'][identity]
        reported = metrics['models'][identity]['evaluation']
        if record['checkpoint_sha256'] != sha(record['checkpoint']) or record['checkpoint_sha256'] != status['checkpoint_sha256']:
            raise ValueError('Current checkpoint binding changed')
        if status['checkpoint_bytes'] != Path(record['checkpoint']).stat().st_size:
            raise ValueError('Reported checkpoint size differs from disk')
        for split, key in [('validation', 'full_validation'), ('test', 'held_out_test')]:
            if not (record[key] == status[split] == reported[split] == plotted[split]):
                raise ValueError('Report/figure/UI metrics differ: '+identity+' '+split)
            csv_row = next(r for r in rows if r['identity'] == identity and r['split'] == split)
            for metric in ['accuracy', 'balanced_accuracy', 'fake_recall', 'f1', 'macro_f1', 'roc_auc', 'loss']:
                if float(csv_row[metric]) != status[split][metric]:
                    raise ValueError('Metrics CSV differs: '+metric)
            path = folder/f'candidate_{split}_fp32.csv'
            if sha(path) != status[split]['predictions_sha256']:
                raise ValueError('Prediction evidence changed')
            if f"{status[split]['accuracy']:.3%}" not in readme:
                raise ValueError('README score is stale')
        training = load(folder/'training_finished.json')
        best_validation = load(folder/'baseline_validation_fp32.json')
        best_epoch = 6 if identity == IDENTITIES[0] else 0
        for row in training['history']:
            if not row.get('historical', False) and better(row['validation'], best_validation):
                best_validation, best_epoch = row['validation'], row['epoch']
        if best_epoch != training['selected_epoch']:
            raise ValueError('Selected epoch is not the validation-selected best')
        expected_start = 1
        epochs = [row['epoch'] for row in training['history']]
        if epochs != list(range(expected_start, training['end_epoch']+1)) or training['end_epoch'] > 10:
            raise ValueError('Training history has missing/duplicate/over-budget epochs')
        if training['selected_epoch'] > training['end_epoch']:
            raise ValueError('Selected epoch was never completed')
        for row in training['history']:
            if not row.get('historical') and row['training_images'] != 100000:
                raise ValueError('New epoch did not use the complete training split')
    if historical_hashes() != preflight['historical_hashes']:
        raise ValueError('Historical video/image evidence changed')
    if any(sha(source['checkpoint']) != source['sha256'] for source in preflight['sources'].values()):
        raise ValueError('A preserved initialization checkpoint changed')
    result = {'passed': True, 'stable_ids': list(IDENTITIES), 'unique_catalog_ids': [r['id'] for r in active],
              'checks': ['checkpoint digests and sizes', 'prediction evidence digests', 'report/CSV/figure/UI metric equality',
                         'README scores', 'validation-only best selection', 'complete epoch history and caps',
                         '100k training counts', 'historical evidence hashes'],
              'historical_files_unchanged': len(preflight['historical_hashes'])}
    atomic_write(root/'final_consistency.json', result)
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--current', action='store_true', help='Verify both stable current image identities')
    parser.add_argument('--promote', action='store_true', help='Activate only after all current-model gates pass')
    parser.add_argument('--check-current', action='store_true', help='Verify final report/figure/catalog consistency from saved evidence')
    args = parser.parse_args()
    if args.check_current:
        check_current_consistency(args.output.resolve())
    elif args.current:
        verify_current_models(args.output.resolve(), args.promote)
    else:
        run(args.output)
