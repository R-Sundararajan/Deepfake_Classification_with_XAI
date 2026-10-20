"""Bounded, reproducible transfer experiments on the existing face-image data."""
import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from tools.runtime import configure_runtime
configure_runtime()
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import DATASET_PATH, MODEL_OUTPUT_PATH
from evaluation.dfdc import binary_metrics, load_baseline
from models.transfer import CLASS_TO_IDX, WEIGHTS, TransferClassifier, load_transfer_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget
from training.dfdc_cache import image_transform

RUN_ROOT=ROOT/'results/execution_20261019'
PREPROCESSING={'size':224,'frames_per_video':8,'interval':1.,'faces':'full_frame','align':False,'margin':1.3,
               'normalization':'ImageNet','color':'RGB','video_faces':'yunet','video_align':False}


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def prepare_manifest(output, train_count=12000, valid_count=2000, seed=42):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[];rng=random.Random(seed);hash_splits={}
    for split,folder,maximum in [('train','train',train_count),('validation','valid',valid_count),('test','test',None)]:
        for label in ['fake','real']:
            files=sorted(p for p in (DATASET_PATH/folder/label).iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'})
            if maximum: files=sorted(rng.sample(files,maximum//2))
            for path in files:
                digest=sha256(path)
                if hash_splits.setdefault(digest,split)!=split: raise ValueError('Exact image duplicate crosses splits: '+str(path))
                records.append({'path':str(path),'label':label.upper(),'target':CLASS_TO_IDX[label],'split':split,'sha256':digest})
    manifest=output/'manifest.csv'
    with manifest.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report={'dataset':'Real vs Fake Faces','manifest_sha256':sha256(manifest),'seed':seed,
            'split_counts':dict(Counter(r['split'] for r in records)),'class_counts':dict(Counter(r['split']+'/'+r['label'] for r in records)),
            'split_policy':'Deterministic balanced subsets of supplied train and valid folders; entire supplied test folder reserved.',
            'exact_hash_overlap_across_selected_splits':False,
            'limitations':'No source-video or identity metadata. Exact hash checks do not rule out identity or near-duplicate leakage. Image classification results do not establish video detection accuracy.'}
    atomic_json(output/'manifest_info.json',report);return report


def load_rows(manifest):
    manifest=Path(manifest);info=json.loads((manifest.parent/'manifest_info.json').read_text(encoding='utf-8'))
    if sha256(manifest)!=info['manifest_sha256']: raise ValueError('Manifest changed')
    with manifest.open(newline='',encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    return rows,info


class ImageRows(Dataset):
    def __init__(self,rows,split,augmentation='none',limit=None):
        self.rows=[r for r in rows if r['split']==split]
        if limit:
            self.rows=sum(([r for r in self.rows if r['label']==label][:limit//2] for label in ['REAL','FAKE']),[])
        self.transform=image_transform(augmentation if split=='train' else 'none')
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['path']) as image: value=self.transform(image.convert('RGB'))
        return value,int(row['target']),index


def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def evaluate(model,dataset,device,batch_size=64):
    model.eval();predictions=[];seconds=0.;loss_sum=0.;started=time.perf_counter()
    loader=DataLoader(dataset,batch_size=batch_size,num_workers=2,persistent_workers=False,pin_memory=device.type=='cuda')
    with torch.inference_mode():
        model(dataset[0][0].unsqueeze(0).to(device));synchronize(device)
        for inputs,targets,indices in loader:
            inputs=inputs.to(device,non_blocking=True);targets=targets.to(device)
            synchronize(device);start=time.perf_counter()
            logits=model(inputs);probabilities=logits.float().softmax(1)[:,0]
            synchronize(device);seconds+=time.perf_counter()-start
            loss_sum+=float(nn.functional.cross_entropy(logits,targets,reduction='sum'))
            for index,p in zip(indices.tolist(),probabilities.cpu().tolist()):
                row=dataset.rows[index]
                predictions.append({'path':row['path'],'label':row['label'],'fake_probability':p,'prediction':'FAKE' if p>=.5 else 'REAL'})
    metrics=binary_metrics([int(r['label']=='FAKE') for r in predictions],[r['fake_probability'] for r in predictions])
    metrics.update(samples=len(predictions),loss=loss_sum/len(dataset),inference_ms_per_image=1000*seconds/len(dataset),evaluation_seconds=time.perf_counter()-started)
    return metrics,predictions


def save_predictions(output,rows):
    with Path(output).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plots(output,metrics,history=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(np.array(metrics['confusion_matrix_real_fake']),display_labels=['REAL','FAKE']).plot(cmap='Blues',colorbar=False)
    plt.title('Real vs Fake Faces / '+Path(output).name);plt.tight_layout();plt.savefig(Path(output)/'confusion_matrix.png',dpi=160);plt.close()
    if history:
        fig,axes=plt.subplots(1,2,figsize=(9,3))
        for key,label in [('training_loss','Train'),('validation_loss','Validation')]: axes[0].plot([h['epoch'] for h in history],[h[key] for h in history],marker='o',label=label)
        axes[0].set(xlabel='Epoch',ylabel='Cross entropy');axes[0].legend()
        axes[1].plot([h['epoch'] for h in history],[h['validation']['roc_auc'] for h in history],marker='o',label='ROC-AUC')
        axes[1].plot([h['epoch'] for h in history],[h['validation']['balanced_accuracy'] for h in history],marker='o',label='Balanced accuracy')
        axes[1].set(xlabel='Epoch',ylabel='Validation metric');axes[1].legend();fig.tight_layout();fig.savefig(Path(output)/'training_curves.png',dpi=160);plt.close(fig)


def train(config_path,smoke=False):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'));seed_all(config.get('seed',42))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',config.get('experiment_id','')):
        raise ValueError('Experiment ID must contain only letters, numbers, underscore or hyphen')
    rows,info=load_rows(config['manifest']);device=torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    output=RUN_ROOT/('smoke_'+config['experiment_id'] if smoke else config['experiment_id']);output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'config.json',config)
    training=ImageRows(rows,'train',config.get('augmentation','basic'),64 if smoke else None)
    validation=ImageRows(rows,'validation',limit=32 if smoke else None)
    # Verify selected file contents before real training; never read test images here.
    print(f"{config['experiment_id']}: verifying {len(training)+len(validation)} selected training/validation images",flush=True)
    for index,row in enumerate(training.rows+validation.rows,1):
        if sha256(row['path'])!=row['sha256']: raise ValueError('Dataset image changed')
        if index%2500==0: print(f'Verified {index} images',flush=True)
    check_budget(512*1024**2)
    model=TransferClassifier(**config['model'],pretrained=True).to(device)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    batch_size=config.get('batch_size',32)
    loader=DataLoader(training,batch_size=batch_size,shuffle=True,num_workers=2,persistent_workers=True,
                      generator=torch.Generator().manual_seed(config.get('seed',42)),pin_memory=device.type=='cuda')
    best=(-1.,-1.,-1.);bad=0;history=[];started=time.perf_counter();current=None
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    phases=[('head',0),('fine_tune',config.get('fine_tune_blocks',1))] if smoke else [('head',0)]*config.get('head_epochs',1)+[('fine_tune',config.get('fine_tune_blocks',1))]*config.get('fine_tune_epochs',2)
    for epoch,(phase,blocks) in enumerate(phases,1):
        if phase!=current:
            if current is not None: model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model_state_dict'])
            model.set_trainable(blocks);current=phase;bad=0
            optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('features.')],'lr':config.get('head_lr',.001)},
                                        {'params':[p for p in model.features.parameters() if p.requires_grad],'lr':config.get('backbone_lr',.0001)}],weight_decay=.0001)
        epoch_start=time.perf_counter();model.train();count=0;loss_sum=0.
        for step,(inputs,targets,_) in enumerate(loader):
            if time.perf_counter()-started>config.get('max_minutes',20)*60: raise RuntimeError('Training time budget exceeded')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(inputs.to(device,non_blocking=True));loss=nn.functional.cross_entropy(logits,targets.to(device))
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach())*len(targets);count+=len(targets)
            if step%100==0: print(f"{config['experiment_id']} epoch {epoch} batch {step+1}/{len(loader)} loss {float(loss.detach()):.4f}",flush=True)
        metrics,predictions=evaluate(model,validation,device,batch_size)
        history.append({'epoch':epoch,'phase':phase,'training_loss':loss_sum/count,'validation_loss':metrics['loss'],
                        'validation':metrics,'epoch_seconds':time.perf_counter()-epoch_start})
        score=tuple(metrics[k] for k in ['roc_auc','balanced_accuracy','f1'])
        payload={'format':'dfdc_transfer_v1','dataset':'Real vs Fake Faces','model_spec':model.spec,'model_state_dict':model.state_dict(),
                 'class_to_idx':CLASS_TO_IDX,'fine_tune_blocks':model.fine_tune_blocks,'image_size':224,'preprocessing':PREPROCESSING,
                 'manifest_sha256':info['manifest_sha256'],'experiment_id':config['experiment_id'],'epoch':epoch,'pretrained_url':WEIGHTS[config['model']['backbone']].url}
        # Unique experiment directories protect all original checkpoints/results.
        torch.save(payload,output/'last.pt')
        if score>best:
            best=score;bad=0;torch.save(payload,output/'best.pt');save_predictions(output/'validation_predictions.csv',predictions)
        else: bad+=1
        atomic_json(output/'history.json',history)
        print(json.dumps({'experiment':config['experiment_id'],'epoch':epoch,'phase':phase,'loss':loss_sum/count,'validation':metrics}),flush=True)
        if bad>=config.get('patience',2): break
    training_seconds=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
    model,saved=load_transfer_checkpoint(output/'best.pt',device)
    metrics,predictions=evaluate(model,validation,device,batch_size)
    save_predictions(output/'validation_predictions.csv',predictions)
    result={'completed':True,'smoke_test':smoke,'dataset':'Real vs Fake Faces','evaluation_unit':'image','evaluation_split':'validation',
            'experiment_id':config['experiment_id'],'model':model.spec,'config':config,'pretrained_url':saved['pretrained_url'],
            'manifest_sha256':info['manifest_sha256'],'training_images':len(training),'validation_images':len(validation),'best_epoch':saved['epoch'],
            'history':history,'validation':metrics,'training_seconds':training_seconds,'gpu_peak_mib':peak,'parameters':sum(p.numel() for p in model.parameters()),
            'checkpoint':str(output/'best.pt'),'checkpoint_sha256':sha256(output/'best.pt'),'checkpoint_bytes':(output/'best.pt').stat().st_size,
            'limitations':info['limitations'],'video_metrics':None,'video_metrics_status':'No labeled video benchmark evaluated'}
    atomic_json(output/'summary.json',result);plots(output,metrics,history);return result


def baseline(manifest):
    seed_all();rows,info=load_rows(manifest);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=RUN_ROOT/'E00_validation';output.mkdir(parents=True,exist_ok=False)
    model=load_baseline(MODEL_OUTPUT_PATH,device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    metrics,predictions=evaluate(model,ImageRows(rows,'validation'),device)
    save_predictions(output/'validation_predictions.csv',predictions)
    result={'completed':True,'experiment_id':'E00','dataset':'Real vs Fake Faces','evaluation_unit':'image','evaluation_split':'validation',
            'model':{'backbone':'resnet18','attention':'none'},'manifest_sha256':info['manifest_sha256'],'validation':metrics,
            'training_seconds':None,'training_time_note':'No retraining; original checkpoint historical training time is separately reported',
            'gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
            'checkpoint':str(MODEL_OUTPUT_PATH),'checkpoint_sha256':sha256(MODEL_OUTPUT_PATH),'checkpoint_bytes':MODEL_OUTPUT_PATH.stat().st_size,
            'limitations':'Legacy checkpoint used full 100,000-image training split and validation for earlier epoch selection. New experiments use the fixed 12,000-image subset; training budgets differ.'}
    atomic_json(output/'summary.json',result);plots(output,metrics);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','train','baseline'])
    parser.add_argument('--config',type=Path);parser.add_argument('--manifest',type=Path,default=RUN_ROOT/'dataset/manifest.csv');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    result=prepare_manifest(args.manifest.parent) if args.action=='prepare' else baseline(args.manifest) if args.action=='baseline' else train(args.config,args.smoke)
    print(json.dumps(result,indent=2),flush=True)
