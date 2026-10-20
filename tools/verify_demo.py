"""Run real model inference and populated Streamlit views on bounded fixtures."""
import argparse
import faulthandler
import io
import json
from pathlib import Path
from unittest.mock import patch

from tools.runtime import configure_runtime
configure_runtime()
import cv2
import numpy as np
import torch
from PIL import Image
from inference.media import MediaPredictor, save_media_result
from streamlit.testing.v1 import AppTest
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json


class Upload(io.BytesIO):
    def __init__(self, path):
        super().__init__(path.read_bytes())
        self.name = path.name
        self.size = path.stat().st_size


def verify(output, device='cuda'):
    output=Path(output); output.mkdir(parents=True, exist_ok=False)
    checkpoint=ROOT/'models/best_resnet18.pth'; before=sha256(checkpoint)
    predictor=MediaPredictor(device=device)
    examples=[]
    for label in ['real','fake']:
        source=next(iter(sorted((ROOT/'data/real_vs_fake/real-vs-fake/test'/label).glob('*.jpg'))))
        with Image.open(source) as image:
            result=predictor.predict_image(image)
        assert np.isfinite(result['explanation']['cam']).all()
        assert abs(result['fake_probability']+result['real_probability']-1)<1e-6
        save_media_result(result,output/('image_'+label))
        examples.append({'source':str(source),'kind':'image','source_label':label,'prediction':result['prediction'],
                         'fake_probability':result['fake_probability'],'elapsed_seconds':result['elapsed_seconds']})
        if label=='real': image_path=source
    # A valid MP4 container made from a held-out face image tests decoding,
    # detection, aggregation and CAM. It supplies no video benchmark accuracy.
    with Image.open(image_path) as image:
        rgb=np.asarray(image.convert('RGB').resize((256,256)))
    video=output/'face_image_smoke_fixture.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,320))
    assert writer.isOpened(), 'MP4 encoder unavailable'
    for index in range(40):
        canvas=np.full((320,320,3),96,dtype=np.uint8)
        offset=24+index%16;canvas[32:288,offset:offset+256]=rgb[:,:,::-1]
        writer.write(canvas)
    writer.release()
    result=predictor.predict_video(video,max_frames=4)
    assert len(result['sampled_frames'])==4 and result['explanations']
    assert any(row['crop_method']=='yunet_largest_face' for row in result['sampled_frames']), result['sampled_frames']
    assert abs(result['fake_probability']-np.mean([r['fake_probability'] for r in result['sampled_frames']]))<1e-7
    save_media_result(result,output/'video')
    examples.append({'kind':'video','source':str(video),'prediction':result['prediction'],
                     'fake_probability':result['fake_probability'],'elapsed_seconds':result['elapsed_seconds'],
                     'sampled_frames':len(result['sampled_frames']),'fixture_only':True})
    blank=Image.fromarray(np.zeros((224,224,3),np.uint8))
    assert predictor._prepare(np.asarray(blank),video=True)[1]=='no_face_full_frame_fallback'
    corrupt=output/'corrupt.mp4';corrupt.write_bytes(b'invalid test container')
    try: predictor.predict_video(corrupt)
    except ValueError: pass
    else: raise AssertionError('Corrupt input accepted')
    # Exercise the actual application branches with upload data, including
    # model loading, metrics, table rendering and explanation images.
    for kind,path in [('Image',image_path),('Video',video)]:
        print('Testing populated Streamlit '+kind+' upload',flush=True)
        faulthandler.dump_traceback_later(50,exit=True)
        app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=45)
        if kind=='Video': app.radio[0].set_value('Video').run(timeout=45)
        with patch('streamlit.file_uploader',return_value=Upload(path)):
            app.run(timeout=60)
        assert not app.exception, str(app.exception)
        assert not app.error, str(app.error)
        assert len(app.metric)==4, f'{kind} upload did not reach inference results'
        if kind=='Video': assert len(app.dataframe)==1
        faulthandler.cancel_dump_traceback_later()
    assert sha256(checkpoint)==before, 'Original checkpoint was modified'
    report={'passed':True,'device':device,'checkpoint_sha256':before,'cases':examples,
            'streamlit_populated_upload_views':['Image','Video'],'corrupt_video_rejected':True,'no_face_fallback_verified':True,
            'limitations':'MP4 fixture consists of one image moving on a canvas. This verifies software operation and is not a labeled video accuracy experiment.'}
    atomic_json(output/'verification.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args();verify(args.output,args.device)
