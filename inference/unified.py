"""Selected frozen research checkpoints, shared image/video protocol and bound XAI."""
import json
import time
import threading
from pathlib import Path

import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image

from evaluation.dfdc import load_baseline
from explainability.gradcam import GradCAM,create_visualizations
from explainability.methods import integrated_gradients,attention_rollout
from explainability.api import compatible_methods, explain as explain_tensor, present
from inference.contracts import checkpoint_digest, class_probabilities, measure
from models.classifier import target_layer
from models.unified import read_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT
from training.dfdc_cache import make_cropper
from training.dfdc_data import probe_video
from training.video_protocol import RUN,SPEC,normalized_frames
from training.unified_experiment import load_encoder,pixels


def available_models():
    """Offer only completed, hash-verified actual configurations."""
    models=[]
    freeze=RUN/'freeze.json'
    if freeze.exists():
        contract=json.loads(freeze.read_text(encoding='utf-8'))
        for id,item in contract['experiments'].items():
            path=Path(item['checkpoint']);summary_path=RUN/id/'summary.json'
            dependencies=['S5','S6'] if id=='S8' else ['S2','S6'] if id=='S7' else ['S2'] if id=='S6' else []
            valid_dependencies=all(Path(contract['experiments'][d]['checkpoint']).exists() and checkpoint_digest(contract['experiments'][d]['checkpoint'])==contract['experiments'][d]['checkpoint_sha256'] for d in dependencies)
            if valid_dependencies and path.exists() and summary_path.exists() and checkpoint_digest(path)==item['checkpoint_sha256'] and checkpoint_digest(summary_path)==item['summary_sha256']:
                summary=json.loads(summary_path.read_text(encoding='utf-8'));models.append({'id':id,**summary,'default':id==contract['champion']})
    return models


class DecisionView(nn.Module):
    """Replace one frame in the selected video decision, retaining other embeddings.

For images, the exact spatial branch is used. Temporal-only image input is an
explicit S2 fallback. A hybrid CAM covers its CNN branch; IG covers both branches.
"""
    def __init__(self,predictor,context=None,position=0,image=False):
        super().__init__();self.predictor=predictor;self.context=context;self.position=position;self.image=image
        # Register the actual modules so zero_grad() also clears selected-head
        # gradients between CAM calls; inference keeps the same module objects.
        if predictor.id=='S0':self.registered_legacy=predictor.legacy
        else:
            self.registered_encoders=nn.ModuleList(predictor.encoders)
            self.registered_spatial=predictor.spatial_head
            if predictor.temporal_head is not None:self.registered_temporal=predictor.temporal_head

    def forward(self,x):
        p=self.predictor
        if p.id=='S0':
            frame=p.legacy(x)[:,[1,0]]
            if self.context is None:return frame
            probabilities=self.context.expand(len(x),-1,-1).clone()
            probabilities[:,self.position]=frame.softmax(-1)
            return probabilities.mean(1).clamp_min(1e-7).log()/getattr(p,'temperature',1.)
        features=torch.cat([model(x) for model in p.encoders],-1).float()
        if self.context is None:
            return p.spatial_head.frame_logits((features-p.spatial_saved['feature_mean'].to(p.device))/p.spatial_saved['feature_std'].to(p.device))
        context=self.context.expand(len(x),-1,-1).clone()
        # Concatenation preserves input gradients without modifying cached context.
        sequence=torch.cat([context[:,:self.position],features[:,None,:],context[:,self.position+1:]],1)
        return p.decision(sequence)/getattr(p,'temperature',1.)


class ResearchPredictor:
    def __init__(self,id,device=None):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        cv2.setNumThreads(2)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lock=threading.RLock()
        options={m['id']:m for m in available_models()}
        if id not in options:raise ValueError('Model is incomplete or checkpoint integrity failed')
        self.id=id;self.summary=options[id];self.config=self.summary['config'];self.status=self.summary['config']['model']
        self.temperature=self.summary['calibration']['temperature'];self.checkpoint_sha256=self.summary['checkpoint_sha256']
        self.crop=make_cropper(SPEC if self.config['preprocess']=='face' else {'faces':'full_frame','align':False})
        self.encoders=[];self.names=[];self.temporal_head=None;self.fusion_weight=None
        if id=='S0':
            self.legacy=load_baseline(ROOT/'models/best_resnet18.pth',self.device);self.layer=target_layer(self.legacy)
            self.methods=compatible_methods();return
        self.image_spatial_id='S5' if id=='S8' else 'S2' if id in {'S6','S7'} else id
        checkpoint=RUN/self.image_spatial_id/'best.pt'
        self.spatial_head,self.spatial_saved=read_checkpoint(checkpoint);self.spatial_head.to(self.device)
        self.names=self.spatial_saved['config']['backbones']
        for name,path,digest in zip(self.names,self.spatial_saved['encoder_paths'],self.spatial_saved['encoder_hashes']):
            if checkpoint_digest(path)!=digest:raise ValueError('Selected encoder hash mismatch')
            self.encoders.append(load_encoder(name,self.device))
        if id in {'S6','S7','S8'}:
            self.temporal_head,self.temporal_saved=read_checkpoint(RUN/'S6/best.pt');self.temporal_head.to(self.device)
            if id in {'S7','S8'}:self.fusion_weight=self.config['spatial_weight']
        self.layer=self.encoders[0].features if self.names[0]=='efficientnet_b0' else None
        self.methods=compatible_methods('efficientnet_b0' in self.names,'deit_tiny_patch16_224' in self.names)

    def decision(self,features,image=False):
        spatial=self.spatial_head((features-self.spatial_saved['feature_mean'].to(self.device))/self.spatial_saved['feature_std'].to(self.device))
        if self.temporal_head is None or image:return spatial
        temporal_features=features[...,:self.temporal_saved['dimension']]
        temporal_input=(temporal_features-self.temporal_saved['feature_mean'].to(self.device))/self.temporal_saved['feature_std'].to(self.device)
        if torch.is_grad_enabled() and temporal_input.requires_grad:
            # Differentiate the same gates without changing process-wide cuDNN
            # flags: independent explanation workers may execute concurrently.
            temporal=self.temporal_head.attribution_forward(temporal_input)
        else:temporal=self.temporal_head(temporal_input)
        if self.id=='S6':return temporal
        p=self.fusion_weight*spatial.softmax(-1)+(1-self.fusion_weight)*temporal.softmax(-1)
        return p.clamp_min(1e-7).log()

    def tensor(self,array):return pixels(array).to(self.device)

    def feature_sequence(self,array):
        cpu,normalization_ms=measure(self.device,lambda:pixels(array))
        x,transfer_ms=measure(self.device,lambda:cpu.to(self.device))
        def forward():
            with torch.inference_mode():
                if self.id=='S0':return self.legacy(x)[:,[1,0]].float().softmax(-1)[None]
                with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                    features=torch.cat([model(x) for model in self.encoders],-1)
                return features.float()[None]
        features,forward_ms=measure(self.device,forward)
        self.feature_timings={'normalization_ms':normalization_ms,'gpu_transfer_ms':transfer_ms,'forward_ms':forward_ms}
        return features

    def score(self,features,image=False):
        with torch.inference_mode():
            logits=features.mean(1).clamp_min(1e-7).log() if self.id=='S0' else self.decision(features,image)
            temperature=1. if image else self.temperature
            p=float(class_probabilities(logits,{'real':0,'fake':1},temperature)['fake'][0])
        return p,logits,temperature

    def attribution_model(self,context=None,position=0,image=False):
        return DecisionView(self,context,position,image).to(self.device).eval()

    def explain(self,image,target,method,context=None,position=0,image_input=False,**parameters):
        if method not in self.methods:raise ValueError('Explanation method incompatible with selected model')
        original=np.asarray(image);tensor=self.tensor(original[None]);view=self.attribution_model(context,position,image_input)
        transformer=self.encoders[self.names.index('deit_tiny_patch16_224')] if 'deit_tiny_patch16_224' in self.names else None
        cam,details=explain_tensor(view,tensor,target,method,self.layer,transformer,**parameters)
        details.update(target_class='FAKE' if target else 'REAL',score_space='selected class logit/log-probability; video temperature included for video decisions',precision='FP32')
        scope='Selected checkpoint decision'
        if self.id in {'S6','S7','S8'} and image_input:scope=f'{self.image_spatial_id} spatial fallback for image input; temporal branch bypassed'
        elif context is not None:scope='Selected complete video decision with other seven frame embeddings held fixed'
        if method in {'gradcam','gradcam++','layercam'} and 'deit_tiny_patch16_224' in self.names:scope+='; CNN branch localization only'
        if method=='attention_rollout':scope='ViT native attention visualization; not class attribution'
        return {**present(original,cam,method,details),'coverage':scope,
                'checkpoint_sha256':checkpoint_digest(RUN/self.image_spatial_id/'best.pt') if image_input and self.id in {'S6','S7','S8'} else self.checkpoint_sha256}

    def predict_image(self,image,method='gradcam',target=None,**parameters):
        start=time.perf_counter();original=np.asarray(image.convert('RGB'))
        (crop,crop_method),crop_ms=measure(self.device,lambda:self.crop(original[:,:,::-1].copy()))
        prepared,resize_ms=measure(self.device,lambda:Image.fromarray(crop[:,:,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR))
        features=self.feature_sequence(np.asarray(prepared)[None])
        (p,_,_),decision_ms=measure(self.device,lambda:self.score(features,True))
        prediction='FAKE' if p>=.5 else 'REAL'
        result={'prediction':prediction,'fake_probability':p,'real_probability':1-p,'confidence':max(p,1-p),
            'confidence_label':'Uncalibrated image score; video temperature is not applied to images',
            'original_input':original,'crop_method':crop_method,'model_status':self.status,
            'prepared_input':np.asarray(prepared),'crop_metadata':getattr(self.crop,'last_metadata',{'bounds_xyxy':[0,0,original.shape[1],original.shape[0]],'face_detection_requested':False}),
            'timings':{'face_detection_ms':crop_ms,'resize_ms':resize_ms,**self.feature_timings,'decision_ms':decision_ms},
            'threshold':.5,'temperature':1.,
            'image_fallback':f'{self.image_spatial_id} spatial checkpoint used; temporal branch bypassed' if self.id in {'S6','S7','S8'} else None,
            'elapsed_seconds':time.perf_counter()-start}
        if method is not None:
            index=int((target or prediction)=='FAKE') if isinstance(target,(str,type(None))) else int(target)
            result['explanation']=self.explain(prepared,index,method,image_input=True,**parameters)
        result['elapsed_seconds']=time.perf_counter()-start
        return result

    def predict_video(self,path,method='gradcam',explanations=3,target=None,**parameters):
        if not 0<=explanations<=5:
            raise ValueError('Use zero to five representative explanations')
        explanation_target=target
        start=time.perf_counter();metadata=probe_video(path);metadata['duration_seconds']=metadata['frame_count']/metadata['fps']
        if metadata['duration_seconds']>7200:raise ValueError('Video exceeds two-hour local input limit')
        arrays=[];rows=[];decode_ms=0.;crop_ms=0.;resize_ms=0.
        frames=iter(normalized_frames(path))
        while True:
            tick=time.perf_counter()
            try:
                index,seconds,bgr=next(frames)
            except StopIteration:
                break
            decode_ms+=(time.perf_counter()-tick)*1000
            (crop,crop_method),elapsed=measure(self.device,lambda:self.crop(bgr));crop_ms+=elapsed
            image,elapsed=measure(self.device,lambda:Image.fromarray(crop[:,:,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR));resize_ms+=elapsed
            arrays.append(np.asarray(image));rows.append({'frame_index':index,'seconds':seconds,'normalized_time':(index+.5)/metadata['frame_count'],'crop_method':crop_method,
                                                         'crop_metadata':getattr(self.crop,'last_metadata',{})})
        features=self.feature_sequence(np.stack(arrays))
        (p,logits,temp),decision_ms=measure(self.device,lambda:self.score(features));target=int(p>=.5)
        with torch.inference_mode():
            if self.id=='S0':framep=features[0,:,1].cpu().numpy()
            else:
                normalized=(features-self.spatial_saved['feature_mean'].to(self.device))/self.spatial_saved['feature_std'].to(self.device)
                framep=self.spatial_head.frame_logits(normalized)[0].softmax(-1)[:,1].cpu().numpy()
        for row,score in zip(rows,framep):row.update({'fake_probability':float(score),'real_probability':float(1-score),'prediction':'FAKE' if score>=.5 else 'REAL'})
        indices=sorted(range(8),key=lambda i:(-framep[i] if target else framep[i],i))[:explanations]
        examples=[]
        contributions={'aggregation':'Arithmetic mean of frame probabilities, then validation temperature'}
        if self.temporal_head is not None:
            with torch.inference_mode():
                spatial=self.spatial_head((features-self.spatial_saved['feature_mean'].to(self.device))/self.spatial_saved['feature_std'].to(self.device)).softmax(-1)[0,1]
                norm=(features[...,:self.temporal_saved['dimension']]-self.temporal_saved['feature_mean'].to(self.device))/self.temporal_saved['feature_std'].to(self.device)
                temporal=self.temporal_head(norm).softmax(-1)[0,1]
                # Leave-one-position replacement at the training feature mean, not padding/extra frames.
                drops=[]
                for i in range(8):
                    perturbed=features.clone();perturbed[:,i,:self.temporal_saved['dimension']]=self.temporal_saved['feature_mean'].to(self.device)
                    changed=self.decision(perturbed).softmax(-1)[0,1]
                    drops.append(float(logits.softmax(-1)[0,1]-changed))
                reversed_score=float(self.temporal_head(norm.flip(1)).softmax(-1)[0,1])
            contributions={'spatial_fake_probability':float(spatial),'temporal_fake_probability':float(temporal),
                'score_space':'Uncalibrated branch scores and decision sensitivity before validation temperature',
                'spatial_weight':self.fusion_weight if self.id in {'S7','S8'} else 0.,'temporal_weight':1-self.fusion_weight if self.id in {'S7','S8'} else 1.,
                'position_mean_replacement_fake_probability_drops':drops,'temporal_reversed_sequence_fake_probability':reversed_score,
                'limitation':'Feature replacement/reversal measures sensitivity; it is not a temporal anomaly detector or causal proof'}
        metadata['sampled_frame_count']=8;metadata['face_status_counts']={m:sum(r['crop_method']==m for r in rows) for m in {r['crop_method'] for r in rows}}
        result={'prediction':'FAKE' if target else 'REAL','fake_probability':p,'real_probability':1-p,'confidence':max(p,1-p),
            'confidence_label':'Validation-temperature model score','model_status':self.status,'metadata':metadata,
            'sampled_frames':rows,'explanations':examples,'aggregation':'Selected system; eight normalized frames',
            'timeline_caption':'Uncalibrated spatial frame probabilities; prediction trace, not temporal anomaly proof. Temporal systems show their explicitly named spatial branch.',
            'contributions':contributions,'sampling_matches_evaluation':True,'temperature':temp,
            'prepared_frames':np.stack(arrays),'feature_context':features.detach().cpu().numpy().copy(),
            'timings':{**self.feature_timings,'decode_ms':decode_ms,'face_detection_ms':crop_ms,'resize_ms':resize_ms,'decision_ms':decision_ms},'threshold':.5,
            'elapsed_seconds':time.perf_counter()-start,'limitations':'Small local DFDC subset. Largest-face selection is not identity tracking; fallback uses full frame. No forensic certainty or exhaustive video coverage.'}
        if method is not None:
            result['explanations']=self.explain_result(result,method,target=explanation_target,frame_indices=sorted(indices),**parameters)
        result['elapsed_seconds']=time.perf_counter()-start
        return result

    def explain_result(self,result,method,target=None,frame_indices=None,**parameters):
        """Reuse deterministic preprocessing; never detect a face again for XAI."""
        index=int((target or result['prediction'])=='FAKE')
        if 'prepared_input' in result:
            return self.explain(Image.fromarray(result['prepared_input']),index,method,image_input=True,**parameters)
        context=torch.from_numpy(result['feature_context']).to(self.device)
        frame_indices=frame_indices if frame_indices is not None else [int(np.argmax([r['fake_probability'] if index else 1-r['fake_probability'] for r in result['sampled_frames']]))]
        if not all(0<=position<len(result['sampled_frames']) for position in frame_indices):
            raise ValueError('Invalid representative frame position')
        return [{'frame_index':result['sampled_frames'][position]['frame_index'],'seconds':result['sampled_frames'][position]['seconds'],
                 **self.explain(Image.fromarray(result['prepared_frames'][position]),index,method,context,position,**parameters)} for position in frame_indices]


if __name__=='__main__':
    import argparse
    from tools.runtime import configure_runtime
    from inference.forensics import image_metadata,video_container
    from inference.media import save_media_result
    configure_runtime()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,help='Completed frozen ID, e.g. S0, S6, S8')
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--image',type=Path);group.add_argument('--video',type=Path)
    parser.add_argument('--method',choices=compatible_methods(True,True))
    parser.add_argument('--device',choices=['cpu','cuda']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();predictor=ResearchPredictor(args.model,args.device)
    method=args.method or predictor.methods[0];source=args.image or args.video;before=sha256(source)
    if args.image:
        metadata=image_metadata(source)
        with Image.open(source) as image:result=predictor.predict_image(image,method)
    else:
        metadata=video_container(source);result=predictor.predict_video(source,method)
    if sha256(source)!=before:raise ValueError('Source file changed during inference')
    result['forensics']={'sha256':before,'source_path':str(source.resolve()),'bytes':source.stat().st_size,**metadata}
    save_media_result(result,args.output)
    print(json.dumps({k:result[k] for k in ['prediction','fake_probability','real_probability','confidence_label']},indent=2))
