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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
