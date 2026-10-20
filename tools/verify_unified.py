"""Execute selected-checkpoint XAI, score parity, CPU and populated Streamlit tests.

Software/XAI checks use validation video media only; no held-out labels select
models or explanation parameters. Run outside the restricted socket sandbox for
Streamlit AppTest on Windows.
"""
import argparse
import csv
import faulthandler
import io
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from tools.runtime import configure_runtime
configure_runtime()
from inference.unified import ResearchPredictor,available_models
from inference.media import save_media_result
from inference.forensics import preserve_upload,image_metadata
from explainability.methods import check_attribution
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT,atomic_json,check_budget
from training.video_protocol import RUN,load_rows


class Upload(io.BytesIO):
    def __init__(self,path):
        super().__init__(path.read_bytes());self.name=path.name;self.size=path.stat().st_size


def verify(ui=True):
    models=available_models()
    if not models:raise RuntimeError('No completed frozen models available')
    out=RUN/'verification';out.mkdir(exist_ok=True)
    image_path=next(iter(sorted((ROOT/'data/real_vs_fake/real-vs-fake/valid/real').glob('*.jpg'))))
    rows=load_rows('validation');row=next(r for r in rows if r['label']=='REAL');video_path=Path(row['path'])
    original_hash=sha256(ROOT/'models/best_resnet18.pth');cases=[];sanity=[]
    # Actual original-file hash, PNG/JPEG decoder validation and persistent bytes.
    preserved,provenance=preserve_upload(image_path.read_bytes(),image_path.name,'Image')
    assert sha256(preserved)==sha256(image_path)==provenance['sha256'];image_metadata(preserved)
    for item in models:
        id=item['id'];predictor=ResearchPredictor(id);expected={}
        with (RUN/id/'validation_predictions.csv').open(newline='',encoding='utf-8') as stream:
            expected={r['filename']:float(r['fake_probability']) for r in csv.DictReader(stream)}
        for method in predictor.methods:
            with Image.open(image_path) as source:image=predictor.predict_image(source,method)
            video=predictor.predict_video(video_path,method,explanations=1)
            assert len(video['sampled_frames'])==8
            assert len({r['frame_index'] for r in video['sampled_frames']})==8
            assert abs(video['fake_probability']-expected[row['filename']])<.003,(id,video['fake_probability'],expected[row['filename']])
            for kind,result in [('image',image),('video',video)]:
                explanation=result['explanation'] if kind=='image' else result['explanations'][0]
                assert np.isfinite(explanation['cam']).all()
                assert abs(result['fake_probability']+result['real_probability']-1)<1e-6
                destination=out/f'{id}_{method}_{kind}'
                if not destination.exists():save_media_result(result,destination)
                cases.append({'id':id,'method':method,'kind':kind,'fake_probability':result['fake_probability'],
                              'finite':True,'nonconstant_map':bool(explanation['cam'].std()>1e-8),
                              'elapsed_seconds':result['elapsed_seconds'],'checkpoint_sha256':explanation['checkpoint_sha256'],
                              'cached_validation_score_difference':abs(video['fake_probability']-expected[row['filename']]) if kind=='video' else None})
            explanation=image['explanation'];tensor=predictor.tensor(explanation['original'][None]);target=int(image['fake_probability']>=.5)
            view=predictor.attribution_model(image=True)
            check=check_attribution(view,tensor,target,explanation['cam'])
            # Flip the model input and compare attribution after unflipping.
            flipped=predictor.explain(Image.fromarray(explanation['original'][:,::-1].copy()),target,method,image_input=True)['cam'][:,::-1]
            cam=explanation['cam'];corr=float(np.corrcoef(cam.ravel(),flipped.ravel())[0,1]) if cam.std()>1e-8 and flipped.std()>1e-8 else None
            sanity.append({'id':id,'method':method,'scope':'one validation image software/stability check; not a population faithfulness estimate',
                           **check,'flip_correlation':corr,'IG':explanation.get('details',{})})
            print(f'verified {id} {method} image/video',flush=True)
        del predictor;torch.cuda.empty_cache()
    cpu=ResearchPredictor(next(m['id'] for m in models if m['default']),'cpu')
    with Image.open(image_path) as source:cpu_image=cpu.predict_image(source,'integrated_gradients')
    cpu_video=cpu.predict_video(video_path,'integrated_gradients',explanations=1)
    assert np.isfinite(cpu_image['fake_probability']) and len(cpu_video['sampled_frames'])==8
    with (RUN/cpu.id/'validation_predictions.csv').open(newline='',encoding='utf-8') as stream:
        cpu_reference=next(float(r['fake_probability']) for r in csv.DictReader(stream) if r['filename']==row['filename'])
    cpu_difference=abs(cpu_video['fake_probability']-cpu_reference)
    assert cpu_difference<.015,('CPU/CUDA score drift',cpu_difference)
    cpu_record={'passed':True,'id':cpu.id,'image_probability':cpu_image['fake_probability'],'video_probability':cpu_video['fake_probability'],
                'image_seconds':cpu_image['elapsed_seconds'],'video_seconds':cpu_video['elapsed_seconds'],'cached_cuda_score_difference':cpu_difference}
    del cpu
    ui_cases=[]
    if ui:
        from streamlit.testing.v1 import AppTest
        # Exercise each actual compatible method through both populated upload branches.
        for item in models:
            for kind,path in [('Image',image_path),('Video',video_path)]:
                app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=90)
                if kind=='Video':app.radio[0].set_value('Video').run(timeout=90)
                app.selectbox[0].set_value(item['id']).run(timeout=90)
                methods=ResearchPredictor(item['id']).methods
                for method in methods:
                    faulthandler.dump_traceback_later(110,exit=True)
                    app.selectbox[1].set_value(method)
                    with patch('streamlit.file_uploader',return_value=Upload(path)):app.run(timeout=100)
                    assert not app.exception,(item['id'],kind,method,str(app.exception))
                    assert not app.error,(item['id'],kind,method,str(app.error))
                    assert len(app.metric)==4
                    if kind=='Video':assert len(app.dataframe)==1
                    ui_cases.append({'id':item['id'],'kind':kind,'method':method,'passed':True})
                    faulthandler.cancel_dump_traceback_later()
                print(f'Streamlit verified {item["id"]} {kind}',flush=True)
    assert sha256(ROOT/'models/best_resnet18.pth')==original_hash
    report={'passed':True,'model_count':len(models),'inference_cases':len(cases),'streamlit_cases':len(ui_cases),'cpu_fallback':cpu_record,
            'original_checkpoint_sha256':original_hash,'inference':cases,'streamlit':ui_cases,'xai_sanity':sanity,
            'media_scope':'Validation DFDC REAL video plus existing validation image; not an accuracy experiment',
            'storage':check_budget()}
    atomic_json(RUN/'verification.json',report);print(json.dumps({k:report[k] for k in ['passed','model_count','inference_cases','streamlit_cases','cpu_fallback']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-ui',action='store_true');args=parser.parse_args();verify(not args.no_ui)
