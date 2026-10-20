"""Shared image/video inference with checkpoint-bound preprocessing and CAMs."""
import argparse
import json
import time
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.dfdc import load_baseline, mean_fake_probability
from explainability.gradcam import GradCAM, create_visualizations
from explainability.methods import integrated_gradients
from explainability.api import compatible_methods, explain as explain_tensor, present
from inference.contracts import class_probabilities, measure
from models.classifier import target_layer
from models.transfer import CLASS_TO_IDX, load_transfer_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT
from training.dfdc_cache import image_transform, make_cropper
from training.dfdc_data import iter_frames, probe_video


class MediaPredictor:
    def __init__(self,checkpoint=None,device=None):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lock=threading.RLock()
        self.methods=compatible_methods()
        champion=ROOT/'results/champion.json'
        if checkpoint is None and champion.exists():
            record=json.loads(champion.read_text(encoding='utf-8'));checkpoint=Path(record['checkpoint'])
            if sha256(checkpoint)!=record['checkpoint_sha256']: raise ValueError('Champion checkpoint hash mismatch')
            self.status=record.get('status', 'Selected model')
        else: self.status='Specified checkpoint' if checkpoint else 'Existing image-trained baseline; DFDC not validated'
        self.path=Path(checkpoint or ROOT/'models/best_resnet18.pth')
        saved=torch.load(self.path,map_location='cpu',weights_only=True)
        self.legacy=saved.get('format')!='dfdc_transfer_v1'
        if self.legacy:
            self.model=load_baseline(self.path,self.device);self.layer=target_layer(self.model)
            self.spec={'size':224,'frames_per_video':4,'interval':1.,'faces':'full_frame','align':False,'margin':1.3}
        else:
            self.model,saved=load_transfer_checkpoint(self.path,self.device);self.layer=self.model.cam_layer;self.spec=saved['preprocessing']
        self.crop=make_cropper(self.spec);self.transform=image_transform('none',224)
        # Preserve full-face image preprocessing for the existing checkpoint.
        # Videos contain scenes, so explicitly detect the largest face first.
        self.video_spec={**self.spec,'faces':'yunet','align':False} if self.legacy else self.spec
        if self.spec.get('video_faces'):
            self.video_spec={**self.spec,'faces':self.spec['video_faces'],'align':self.spec.get('video_align',False)}
        self.video_crop=make_cropper(self.video_spec)
        self.checkpoint_sha256=sha256(self.path)

    def _prepare(self,bgr,video=False):
        crop,method=(self.video_crop if video else self.crop)(bgr)
        image=Image.fromarray(crop[:,:,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR)
        return image,method

    def _probability(self,image):
        with torch.inference_mode(): return float(self.model(self.transform(image).unsqueeze(0).to(self.device)).float().softmax(1)[0,0])

    def explain(self,image,predicted_index,method='gradcam',**parameters):
        tensor=self.transform(image).unsqueeze(0).to(self.device)
        cam,details=explain_tensor(self.model,tensor,predicted_index,method,self.layer,**parameters)
        details.update(target_class='FAKE' if predicted_index==0 else 'REAL',precision='FP32')
        return {**present(np.asarray(image),cam,method,details),'checkpoint_sha256':self.checkpoint_sha256,
                'coverage':'CNN spatial branch only; the frequency descriptor is not spatially localized' if not self.legacy and self.model.frequency is not None else 'CNN spatial features'}

    def predict_image(self,image,method='gradcam',target=None,**parameters):
        start=time.perf_counter();input_rgb=np.asarray(image.convert('RGB'))
        (crop,crop_method),crop_ms=measure(self.device,lambda:self.crop(input_rgb[:,:,::-1].copy()))
        prepared,resize_ms=measure(self.device,lambda:Image.fromarray(crop[:,:,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR))
        cpu,normalization_ms=measure(self.device,lambda:self.transform(prepared).unsqueeze(0))
        tensor,transfer_ms=measure(self.device,lambda:cpu.to(self.device))
        def forward():
            with torch.inference_mode():
                return float(class_probabilities(self.model(tensor),CLASS_TO_IDX)['fake'][0])
        p,forward_ms=measure(self.device,forward)
        index=0 if p>=.5 else 1
        result={'prediction':'FAKE' if index==0 else 'REAL','fake_probability':p,'real_probability':1-p,'confidence':max(p,1-p),
                'confidence_label':'Uncalibrated model score','model_status':self.status,'crop_method':crop_method,
                'original_input':input_rgb,'prepared_input':np.asarray(prepared),'threshold':.5,'temperature':1.,
                'crop_metadata':getattr(self.crop,'last_metadata',{'bounds_xyxy':[0,0,input_rgb.shape[1],input_rgb.shape[0]],'face_detection_requested':False}),
                'timings':{'face_detection_ms':crop_ms,'resize_ms':resize_ms,'normalization_ms':normalization_ms,'gpu_transfer_ms':transfer_ms,'forward_ms':forward_ms},
                'elapsed_seconds':time.perf_counter()-start}
        if method is not None:
            result['explanation']=self.explain(prepared,CLASS_TO_IDX[(target or result['prediction']).lower()],method,**parameters)
        result['elapsed_seconds']=time.perf_counter()-start
        return result

    def explain_result(self,result,method,target=None,**parameters):
        if 'prepared_input' not in result:
            raise ValueError('These image-trained checkpoints require image mode in the demo')
        return self.explain(Image.fromarray(result['prepared_input']),CLASS_TO_IDX[(target or result['prediction']).lower()],method,**parameters)

    def predict_video(self,path,method='gradcam',explanations=3,max_frames=None):
        start=time.perf_counter();path=Path(path)
        if not path.is_file() or path.stat().st_size>2*1024**3: raise ValueError('Missing video or video exceeds 2 GiB local input limit')
        if not 0<=explanations<=5: raise ValueError('Use 0..5 representative explanations')
        metadata=probe_video(path)
        if metadata['frame_count']/metadata['fps']>7200: raise ValueError('Video duration exceeds two hours')
        count=max_frames if max_frames is not None else self.spec['frames_per_video']
        rows=[]
        for index,seconds,bgr in iter_frames(path,self.spec['interval'],count):
            image,crop_method=self._prepare(bgr,video=True);p=self._probability(image)
            rows.append({'frame_index':index,'seconds':seconds,'prediction':'FAKE' if p>=.5 else 'REAL',
                         'fake_probability':p,'real_probability':1-p,'confidence':max(p,1-p),'crop_method':crop_method})
        p=mean_fake_probability([r['fake_probability'] for r in rows]);prediction='FAKE' if p>=.5 else 'REAL';class_index=CLASS_TO_IDX[prediction.lower()]
        representative=sorted(rows,key=lambda r:(-r['fake_probability'] if prediction=='FAKE' else r['fake_probability'],r['frame_index']))[:explanations]
        selected={r['frame_index'] for r in representative};examples=[]
        # Re-decode selected frames; do not retain full-resolution videos in RAM.
        for index,seconds,bgr in iter_frames(path,self.spec['interval'],count):
            if index in selected:
                image,_=self._prepare(bgr,video=True);examples.append({'frame_index':index,'seconds':seconds,**self.explain(image,class_index,method)})
        return {'prediction':prediction,'fake_probability':p,'real_probability':1-p,'confidence':max(p,1-p),'confidence_label':'Uncalibrated model score',
                'model_status':self.status,'metadata':metadata,'sampled_frames':rows,'explanations':examples,'aggregation':'Arithmetic mean of frame FAKE probabilities',
                'sampling_matches_evaluation':not self.legacy and count==self.spec['frames_per_video'],'elapsed_seconds':time.perf_counter()-start,
                'video_preprocessing':self.video_spec,
                'limitations':'A limited deterministic sample does not represent every moment or face. Largest-face selection does not track identities. Heatmaps show model influence, not proof.'}


def save_media_result(result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    def serialize(value,prefix):
        if isinstance(value,np.ndarray):
            if value.ndim==3 and value.shape[-1] in {3,4} and value.dtype==np.uint8:
                file=output/(prefix+'.png');Image.fromarray(value).save(file)
            else:
                file=output/(prefix+'.npy');np.save(file,value)
            return str(file)
        if isinstance(value,dict): return {k:serialize(v,prefix+'_'+k) for k,v in value.items()}
        if isinstance(value,list): return [serialize(v,prefix+'_'+str(i)) for i,v in enumerate(value)]
        return value
    serializable=serialize(result,'prediction')
    (output/'prediction.json').write_text(json.dumps(serializable,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image',type=Path);group.add_argument('--video',type=Path)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=compatible_methods(),default='gradcam');parser.add_argument('--frames',type=int)
    args=parser.parse_args()
    from tools.runtime import configure_runtime
    configure_runtime();torch.set_num_threads(2)
    predictor=MediaPredictor(args.checkpoint)
    if args.image:
        with Image.open(args.image) as image: result=predictor.predict_image(image,args.method)
    else: result=predictor.predict_video(args.video,args.method,max_frames=args.frames)
    save_media_result(result,args.output)
    print(json.dumps({k:result[k] for k in ['prediction','fake_probability','confidence_label','model_status','elapsed_seconds']},indent=2))
