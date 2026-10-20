"""Cross-dataset evaluation of the existing ResNet-18; no training or downloads."""
import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

# Required by deterministic CUDA linear algebra; set before context creation.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from PIL import Image

from config import IMAGENET_MEAN,IMAGENET_STD
from models.classifier import create_model
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT,atomic_json,check_budget,owned_path
from training.dfdc_data import FaceCropper,iter_frames


def mean_fake_probability(probabilities):
    a=np.asarray(probabilities,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Require a nonempty vector of finite probabilities in [0,1]')
    return float(a.mean())


def binary_metrics(labels,probabilities,threshold=.5):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,log_loss,balanced_accuracy_score,average_precision_score,brier_score_loss
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p) or not len(y) or not set(y).issubset({0,1}):raise ValueError('Invalid binary labels/probability lengths')
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Threshold must be in [0,1]')
    y=y.astype(int)
    mean_fake_probability(p)
    pred=(p>=threshold).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),
            'real_precision':float(precision_score(y,pred,pos_label=0,zero_division=0)),
            'real_recall':float(recall_score(y,pred,pos_label=0,zero_division=0)),
            'fake_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),
            'macro_f1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
            'pr_auc':float(average_precision_score(y,p)) if len(set(y))==2 else None,
            'brier_score':float(brier_score_loss(y,p)),
            'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if len(set(y))==2 else None,
            'log_loss':float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
            'confusion_matrix_real_fake':confusion_matrix(y,pred,labels=[0,1]).tolist(),
            'positive_class':'FAKE','threshold':threshold,'undefined_precision_recall_f1_policy':'zero when denominator zero'}


def load_baseline(checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('model_name')!='resnet18' or saved.get('class_to_idx')!={'fake':0,'real':1} or saved.get('image_size')!=224:
        raise ValueError('Checkpoint model/label/resolution contract differs from verified existing prototype')
    model=create_model(pretrained=False,trainable_backbone_blocks=int(saved.get('trainable_backbone_blocks',1)))
    model.load_state_dict(saved['model_state_dict'])
    return model.to(device).eval()


def preprocessing_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])


def frame_probability(model,frame,transform,device):
    tensor=transform(Image.fromarray(frame[:,:,::-1].copy())).unsqueeze(0).to(device)
    return float(torch.softmax(model(tensor),dim=1)[0,0].item())


def run(args):
    start=time.perf_counter();manifest=args.manifest.resolve()
    preparation=json.loads((manifest.parent/'preparation.json').read_text(encoding='utf-8'))
    if sha256(manifest)!=preparation['manifest_sha256']:raise ValueError('Manifest hash changed since preparation')
    if not preparation['official_provenance_verified']:raise ValueError('Real baseline requires verified official Kaggle sample provenance')
    with manifest.open(newline='',encoding='utf-8') as f:records=list(csv.DictReader(f))
    selected=[r for r in records if args.split=='all' or r['split']==args.split]
    if not selected:raise ValueError('No selected videos')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    checkpoint=args.checkpoint.resolve();before_hash=sha256(checkpoint)
    # Existing trusted local checkpoint, weights-only loading; no pretrained download.
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
        free,_=torch.cuda.mem_get_info(device)
        if free<512*1024*1024:raise RuntimeError(f'Insufficient free GPU memory: {free} bytes')
    model=load_baseline(checkpoint,device)
    transform=preprocessing_transform()
    crop=FaceCropper(args.faces)
    run_id=args.run_id or datetime.now(timezone.utc).strftime('resnet18_cross_%Y%m%dT%H%M%SZ')
    if not run_id.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe run ID')
    out=owned_path(ROOT/'results/dfdc_runs'/run_id)
    if out.exists():raise FileExistsError('Experiment directory exists; choose a new run ID')
    check_budget(len(selected)*(args.max_frames*400+3000)+1024*1024);out.mkdir(parents=True)
    videos=[];errors=[];crop_counts=Counter();frames=0
    with (out/'frame_predictions.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['filename','frame_index','seconds','fake_probability','preprocessing'])
        for i,row in enumerate(selected,1):
            if row['status']!='valid':errors.append({'filename':row['filename'],'error':'Invalid at preparation'});continue
            probabilities=[];local_crops=Counter()
            try:
                p=Path(row['path'])
                if sha256(p)!=row['sha256']:raise ValueError('Video hash differs from manifest')
                with torch.inference_mode():
                    for index,seconds,frame in iter_frames(p,args.interval,args.max_frames):
                        face,method=crop(frame)
                        prob=frame_probability(model,face,transform,device)
                        probabilities.append(prob);local_crops[method]+=1
                        writer.writerow([row['filename'],index,seconds,prob,method]);frames+=1
                prob=mean_fake_probability(probabilities)
                videos.append({'filename':row['filename'],'label':row['label'],'source_group':row['source_group'],'split':row['split'],'frames':len(probabilities),'fake_probability':prob,'predicted_label':'FAKE' if prob>=.5 else 'REAL'})
                crop_counts.update(local_crops)
            except (ValueError,OSError,RuntimeError) as e:
                errors.append({'filename':row['filename'],'error':str(e)[:250]})
                # Partial-frame predictions remain in CSV for audit, but the whole
                # failed video is excluded from all video metrics.
            if i%10==0 or i==len(selected):print(f'Evaluated {i}/{len(selected)}; successful {len(videos)}',flush=True);check_budget()
    if not videos:
        atomic_json(out/'failure.json',{'completed':False,'errors':errors});raise RuntimeError('No videos evaluated; metrics not produced')
    if device.type=='cuda':torch.cuda.synchronize(device)
    metrics=binary_metrics([int(r['label']=='FAKE') for r in videos],[r['fake_probability'] for r in videos])
    with (out/'video_predictions.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(videos[0]));w.writeheader();w.writerows(videos)
    if sha256(checkpoint)!=before_hash:raise RuntimeError('Checkpoint changed during evaluation')
    limitations='Small official training sample, not official hidden competition test. Cross-dataset only; no DFDC training/tuning. Source groups do not guarantee actor disjointness. '+('Full frames: no face detection/alignment; background and face scale differ from the face-image training domain. ' if args.faces=='full_frame' else 'Haar largest-face crop has no identity tracking/alignment; no-face frames use full-frame fallback. ')+'No confidence intervals; class imbalance affects accuracy.'
    summary={'completed':True,'dataset_provenance':'official_kaggle_dfdc','experiment_id':run_id,'model':'Existing ResNet-18',
             'evaluation_dataset':f'Official DFDC labeled training sample / {args.split} generated split(s)',
             'manifest':str(manifest),'manifest_sha256':sha256(manifest),'checkpoint':str(checkpoint),'checkpoint_sha256':before_hash,
             'videos_selected':len(selected),'videos_evaluated':len(videos),'frames_evaluated':sum(v['frames'] for v in videos),
             'frames_attempted_including_failed_videos':frames,'coverage':len(videos)/len(selected),'errors':errors,
             'label_counts':dict(Counter(v['label'] for v in videos)),'source_groups_evaluated':len({v['source_group'] for v in videos}),
             'preprocessing':('Full frames; no face detection/alignment' if args.faces=='full_frame' else 'Largest Haar face, margin1.3, detection max side640; full-frame fallback')+'; RGB Resize224 ImageNet normalization',
             'preprocessing_frame_counts':dict(crop_counts),'frame_interval_seconds':args.interval,'max_frames_per_video':args.max_frames,
             'aggregation':'Arithmetic mean of sampled frame FAKE probabilities; FAKE output index 0; video threshold >= .5',
             'metrics':metrics,'gpu_peak_mib':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
             'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
             'elapsed_seconds':time.perf_counter()-start,'training_seconds':0,'seed':args.seed,
             'torch_version':torch.__version__,'numpy_version':np.__version__,'limitations':limitations,'storage':check_budget()}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/dfdc/prepared_seed42/manifest.csv')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/best_resnet18.pth')
    p.add_argument('--split',choices=['train','validation','test','all'],default='test')
    p.add_argument('--faces',choices=['haar','full_frame'],default='full_frame')
    p.add_argument('--interval',type=float,default=1.0);p.add_argument('--max-frames',type=int,default=16)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-id');run(p.parse_args())
