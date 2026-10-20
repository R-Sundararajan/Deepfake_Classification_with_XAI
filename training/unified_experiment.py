"""Sequential, resumable frozen-feature DFDC experiments with sealed test release.

Commands: features, train, freeze, test. Test predictions are prohibited before
freeze.json exists and are claimed once per experiment before computation.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as TF

from evaluation.dfdc import binary_metrics,load_baseline
from models.unified import DIMENSIONS,FeatureHead,encoder,read_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT,atomic_json,check_budget
from training.video_protocol import RUN,CACHE,SPEC,load_rows,write_csv

FEATURES=ROOT/'data/dfdc/runtime/unified_features'
ENCODERS=ROOT/'data/dfdc/runtime/unified_encoders'
CONFIGS=[
    {'id':'S0','model':'Preserved image ResNet-18','backbones':[],'preprocess':'raw','balance':False,'augment':False,'kind':'legacy'},
    {'id':'S1','model':'EfficientNet-B0','backbones':['efficientnet_b0'],'preprocess':'raw','balance':False,'augment':False,'kind':'spatial'},
    {'id':'S1p','model':'EfficientNet-B0','backbones':['efficientnet_b0'],'preprocess':'face','balance':False,'augment':False,'kind':'spatial'},
    {'id':'S1b','model':'EfficientNet-B0','backbones':['efficientnet_b0'],'preprocess':'face','balance':True,'augment':False,'kind':'spatial'},
    {'id':'S2','model':'EfficientNet-B0','backbones':['efficientnet_b0'],'preprocess':'face','balance':True,'augment':True,'kind':'spatial'},
    {'id':'S3','model':'DeiT-Tiny','backbones':['deit_tiny_patch16_224'],'preprocess':'raw','balance':False,'augment':False,'kind':'spatial'},
    {'id':'S4','model':'DeiT-Tiny','backbones':['deit_tiny_patch16_224'],'preprocess':'face','balance':True,'augment':True,'kind':'spatial'},
    {'id':'S5','model':'EfficientNet-B0 + DeiT-Tiny','backbones':['efficientnet_b0','deit_tiny_patch16_224'],'preprocess':'face','balance':True,'augment':True,'kind':'spatial'},
    {'id':'S6','model':'EfficientNet-B0 + one-layer LSTM','backbones':['efficientnet_b0'],'preprocess':'face','balance':True,'augment':True,'kind':'temporal'},
]


def setup():
    from tools.runtime import configure_runtime
    configure_runtime();torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('Training protocol requires the verified CUDA device')
    return torch.device('cuda')


def pixels(array, augmentation=False, seed=42):
    x=torch.from_numpy(array.copy()).permute(0,3,1,2).float()/255
    if augmentation:
        # One reproducible extra view per training video, shared across frames.
        # Weighted sampling remains at video level; frame order/count are fixed.
        rng=random.Random(seed)
        if rng.random()<.5:x=x.flip(-1)
        x=TF.adjust_brightness(x,rng.uniform(.85,1.15))
        x=TF.adjust_contrast(x,rng.uniform(.85,1.15))
        x=TF.adjust_saturation(x,rng.uniform(.85,1.15))
    return (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]


def encoder_path(name):return ENCODERS/(name+'.pt')


def load_encoder(name,device):
    model=encoder(name,False)
    model.load_state_dict(torch.load(encoder_path(name),map_location='cpu',weights_only=True)['state_dict'])
    return model.to(device).eval()


def extract(split,device):
    if split=='test' and not (RUN/'freeze.json').exists():raise ValueError('Test extraction requires frozen configurations')
    FEATURES.mkdir(parents=True,exist_ok=True);ENCODERS.mkdir(parents=True,exist_ok=True)
    rows=load_rows(split);audit=json.loads((RUN/'audit.json').read_text(encoding='utf-8'))
    for name in DIMENSIONS:
        if not encoder_path(name).exists():
            model=encoder(name,True)
            torch.save({'state_dict':model.state_dict(),'source':'ImageNet-1K pretrained','architecture':name},encoder_path(name))
            del model
        model=load_encoder(name,device);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        # Profile a real eight-frame batch before the full extraction.
        with np.load(CACHE/(Path(rows[0]['filename']).stem+'.npz')) as source:
            smoke=pixels(source['raw']).to(device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):smoke_features=model(smoke)
        if not torch.isfinite(smoke_features).all():raise RuntimeError('Nonfinite smoke features')
        atomic_json(RUN/f'smoke_{name}_{split}.json',{'passed':True,'batch':8,'shape':list(smoke_features.shape),'peak_vram_bytes':torch.cuda.max_memory_allocated()})
        for variant in ['raw','face','face_aug'] if split=='train' else ['raw','face']:
            dest=FEATURES/f'{name}_{variant}_{split}.pt'
            if dest.exists():continue
            values=[];model_seconds=0
            for i,row in enumerate(rows):
                with np.load(CACHE/(Path(row['filename']).stem+'.npz'),allow_pickle=False) as source:
                    tensor=pixels(source['raw' if variant=='raw' else 'face'],variant=='face_aug',42+i).to(device)
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):features=model(tensor)
                torch.cuda.synchronize();model_seconds+=time.perf_counter()-tick
                values.append(features.float().cpu())
                if (i+1)%100==0:print(f'features {name} {variant} {split} {i+1}/{len(rows)}',flush=True)
            torch.save({'values':torch.stack(values),'filenames':[r['filename'] for r in rows],
                        'manifest_sha256':audit['manifest_sha256'],'encoder_sha256':sha256(encoder_path(name))},dest)
            atomic_json(RUN/f'features_{name}_{variant}_{split}.json',{'videos':len(rows),'frames':len(rows)*8,
                'forward_seconds':model_seconds,'forward_ms_per_video':model_seconds/len(rows)*1000,
                'peak_vram_bytes':torch.cuda.max_memory_allocated(),'encoder_sha256':sha256(encoder_path(name))})
        del model;torch.cuda.empty_cache()
        print(f'encoder {name} {split} done {time.perf_counter()-started:.1f}s',flush=True)
    if split=='validation':legacy_predictions(split,device)


def feature_data(config,split,augment=False):
    rows=load_rows(split);arrays=[]
    variant='face_aug' if augment else config['preprocess']
    for name in config['backbones']:
        saved=torch.load(FEATURES/f'{name}_{variant}_{split}.pt',map_location='cpu',weights_only=True)
        if saved['filenames']!=[r['filename'] for r in rows]:raise ValueError('Feature ordering mismatch')
        if saved['manifest_sha256']!=sha256(RUN/'manifest.csv') or saved['encoder_sha256']!=sha256(encoder_path(name)):raise ValueError('Stale feature cache')
        arrays.append(saved['values'])
    return torch.cat(arrays,-1),torch.tensor([int(r['label']=='FAKE') for r in rows]),rows


def probability(logits,temperature=1.):
    return torch.softmax(torch.as_tensor(logits,dtype=torch.float64)/temperature,-1)[:,1].numpy()


def ece(y,p,bins=10):
    y=np.asarray(y);p=np.asarray(p);confidence=np.maximum(p,1-p);correct=(p>=.5)==y;total=0.
    for lo in np.linspace(.5,1,bins,endpoint=False):
        mask=(confidence>=lo)&(confidence<lo+.5/bins if lo+.5/bins<1 else confidence<=1)
        if mask.any():total+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return float(total)


def calibrate(y,logits):
    from scipy.optimize import minimize_scalar
    y=np.asarray(y)
    def loss(logtemp):
        p=np.clip(probability(logits,float(np.exp(logtemp))),1e-7,1-1e-7)
        return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
    fit=minimize_scalar(loss,bounds=(-2.3,3),method='bounded')
    temperature=float(np.exp(fit.x));before=probability(logits);after=probability(logits,temperature)
    return {'temperature':temperature,'fit_split':'validation','method':'scalar temperature on video aggregate log probabilities/logits',
            'before':{**binary_metrics(y,before),'ece':ece(y,before)},'after':{**binary_metrics(y,after),'ece':ece(y,after)},
            'inconclusive_enabled':False,'inconclusive_reason':'Small REAL validation sample cannot establish a reliable selective-risk guarantee; no threshold deployed'}


def save_predictions(path,rows,logits,temperature=1.):
    p=probability(logits,temperature)
    write_csv(path,[{'filename':row['filename'],'source_group':row['source_group'],'label':row['label'],
                    'fake_probability':float(score),'real_probability':float(1-score),
                    'raw_fake_probability':float(raw)} for row,score,raw in zip(rows,p,probability(logits))])


def legacy_predictions(split,device):
    out=RUN/'S0';out.mkdir(exist_ok=True)
    if (out/f'{split}_logits.pt').exists():return
    if split=='test' and not (RUN/'freeze.json').exists():raise ValueError('Test is sealed')
    rows=load_rows(split);model=load_baseline(ROOT/'models/best_resnet18.pth',device);values=[]
    started=time.perf_counter();forward=0;torch.cuda.reset_peak_memory_stats()
    for row in rows:
        with np.load(CACHE/(Path(row['filename']).stem+'.npz')) as source:tensor=pixels(source['raw']).to(device)
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.inference_mode():p=model(tensor).float().softmax(-1)[:,0].mean().clamp(1e-7,1-1e-7)
        torch.cuda.synchronize();forward+=time.perf_counter()-tick
        values.append(torch.stack([(1-p).log(),p.log()]).cpu())
    logits=torch.stack(values);torch.save(logits,out/f'{split}_logits.pt')
    if split=='validation':
        calibration=calibrate([int(r['label']=='FAKE') for r in rows],logits)
        summary={'id':'S0','config':CONFIGS[0],'validation':calibration['before'],'calibration':calibration,
            'training_seconds':0,'training_note':'Preserved image-trained checkpoint, no DFDC training',
            'inference_ms_per_video':forward/len(rows)*1000,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
            'checkpoint':str(ROOT/'models/best_resnet18.pth'),'checkpoint_sha256':sha256(ROOT/'models/best_resnet18.pth'),
            'validation_videos':len(rows),'elapsed_seconds':time.perf_counter()-started}
        atomic_json(out/'summary.json',summary);save_predictions(out/'validation_predictions.csv',rows,logits,calibration['temperature'])
    del model;torch.cuda.empty_cache()


def train_one(config,device):
    out=RUN/config['id'];out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text(encoding='utf-8'))
    x,y,trainrows=feature_data(config,'train');v,vy,valrows=feature_data(config,'validation')
    ax=feature_data(config,'train',True)[0] if config['augment'] else x
    mean=x.mean((0,1));std=x.std((0,1)).clamp_min(.05)
    x=((x-mean)/std).to(device);ax=((ax-mean)/std).to(device);v=((v-mean)/std).to(device);y=y.to(device)
    torch.manual_seed(42);head=FeatureHead(x.shape[-1],config['kind']).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');best=None;history=[];start=0;stale=0;elapsed=0.
    cfg={**config,'seed':42,'max_epochs':24,'patience':5,'batch_videos':32,'optimizer':'AdamW',
         'lr':.002,'weight_decay':.01,'AMP':True,'encoders':'frozen ImageNet-1K','train_unit':'whole eight-frame video',
         'sampling':'inverse-class-frequency video sampler, replacement, one training-set length per epoch' if config['balance'] else 'natural shuffled videos',
         'augmentation':'one deterministic flip/brightness/contrast/saturation view per training video, randomly original/view each epoch' if config['augment'] else 'none',
         'preprocessing_spec':SPEC,'manifest_sha256':sha256(RUN/'manifest.csv')}
    atomic_json(out/'config.json',cfg)
    if (out/'last.pt').exists():
        last=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if last['config']!=cfg:raise ValueError('Cannot resume a changed experiment')
        head.load_state_dict(last['head']);optimizer.load_state_dict(last['optimizer']);scaler.load_state_dict(last['scaler'])
        history=last['history'];best=tuple(last['best']);start=last['epoch']+1;stale=last['stale'];elapsed=last['elapsed']
    torch.cuda.reset_peak_memory_stats()
    # Real forward/backward smoke before the complete budget.
    optimizer.zero_grad();loss=nn.functional.nll_loss(head(x[:2]),y[:2]);loss.backward();optimizer.zero_grad()
    if not torch.isfinite(loss):raise RuntimeError('Training smoke nonfinite')
    atomic_json(out/'smoke.json',{'passed':True,'batch_videos':2,'sequence_length':8,'loss':float(loss.detach()),'peak_vram_bytes':torch.cuda.max_memory_allocated()})
    weights=1/torch.bincount(y,minlength=2).float()[y]
    for epoch in range(start,24):
        tick=time.perf_counter();torch.manual_seed(42+epoch)
        order=torch.multinomial(weights,len(y),replacement=True) if config['balance'] else torch.randperm(len(y),device=device)
        head.train();losses=[]
        for indices in order.split(32):
            batch=x[indices]
            if config['augment']:
                select=torch.rand(len(indices),device=device)<.5
                batch=torch.where(select[:,None,None],ax[indices],batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=head(batch);loss=nn.functional.cross_entropy(logits,y[indices])
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(head.parameters(),5.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():vl=head(v).float().cpu()
        metrics=binary_metrics(vy.numpy(),probability(vl));score=(metrics['balanced_accuracy'],metrics['macro_f1'],metrics['roc_auc'])
        seconds=time.perf_counter()-tick;elapsed+=seconds
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':seconds,'validation':metrics})
        if best is None or score>best:
            best=score;stale=0
            torch.save({'format':'unified_dfdc_v1','class_to_idx':{'real':0,'fake':1},'config':cfg,'kind':config['kind'],
                        'dimension':x.shape[-1],'head_state':{k:t.detach().cpu() for k,t in head.state_dict().items()},
                        'feature_mean':mean,'feature_std':std,'epoch':epoch+1,
                        'encoder_paths':[str(encoder_path(n)) for n in config['backbones']],
                        'encoder_hashes':[sha256(encoder_path(n)) for n in config['backbones']]},out/'best.pt')
        else:stale+=1
        torch.save({'epoch':epoch,'config':cfg,'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'history':history,'best':list(best),'stale':stale,'elapsed':elapsed},out/'last.pt')
        atomic_json(out/'history.json',history)
        print(f"{config['id']} epoch {epoch+1} bacc={metrics['balanced_accuracy']:.4f} macroF1={metrics['macro_f1']:.4f}",flush=True)
        if stale>=5:break
    selected,saved=read_checkpoint(out/'best.pt');selected=selected.to(device)
    with torch.inference_mode():logits=selected(v).cpu()
    torch.save(logits,out/'validation_logits.pt');calibration=calibrate(vy.numpy(),logits)
    # Batch-one feature-head timing; encoder time measured separately on same validation cohort.
    for _ in range(5):selected(v[:1])
    torch.cuda.synchronize();tick=time.perf_counter()
    with torch.inference_mode():
        for _ in range(50):selected(v[:1])
    torch.cuda.synchronize();head_ms=(time.perf_counter()-tick)/50*1000
    encoder_ms=sum(json.loads((RUN/f"features_{name}_{config['preprocess']}_validation.json").read_text(encoding='utf-8'))['forward_ms_per_video'] for name in config['backbones'])
    summary={'id':config['id'],'config':cfg,'validation':calibration['before'],'calibration':calibration,
             'training_seconds':elapsed,'training_cost_scope':'head training only; shared encoder/cache extraction reported separately',
             'epochs':len(history),'best_epoch':saved['epoch'],'inference_ms_per_video':encoder_ms+head_ms,
             'latency_scope':'encoder batch-eight forward plus batch-one head; excludes decode/cropping/XAI',
             'peak_vram_bytes':torch.cuda.max_memory_allocated(),'checkpoint':str(out/'best.pt'),
             'checkpoint_sha256':sha256(out/'best.pt'),'train_videos':len(trainrows),'validation_videos':len(valrows),
             'trainable_parameters':sum(p.numel() for p in selected.parameters()),'attention':'native ViT self-attention' if len(config['backbones'])>1 or config['backbones'][0].startswith('deit') else 'native EfficientNet squeeze/excitation; no added ECA/CBAM'}
    atomic_json(out/'summary.json',summary);save_predictions(out/'validation_predictions.csv',valrows,logits,calibration['temperature'])
    return summary


def fusion(id='S7',spatial_id='S2'):
    out=RUN/id;out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    spatial=json.loads((RUN/spatial_id/'summary.json').read_text(encoding='utf-8'));temporal=json.loads((RUN/'S6/summary.json').read_text(encoding='utf-8'))
    a=torch.load(RUN/spatial_id/'validation_logits.pt',weights_only=True);b=torch.load(RUN/'S6/validation_logits.pt',weights_only=True)
    rows=load_rows('validation');y=np.array([int(r['label']=='FAKE') for r in rows]);candidates=[]
    for weight in [.25,.5,.75]:
        p=weight*probability(a)+(1-weight)*probability(b);m=binary_metrics(y,p)
        candidates.append((m['balanced_accuracy'],m['macro_f1'],weight,p))
    _,_,weight,p=max(candidates,key=lambda item:item[:2]);logits=torch.tensor(np.column_stack([1-p,p])).log()
    calibration=calibrate(y,logits);torch.save(logits,out/'validation_logits.pt')
    config={'id':id,'model':('CNN–ViT hybrid + LSTM combined system' if spatial_id=='S5' else 'EfficientNet spatial + LSTM late fusion'),'kind':'score_fusion','components':[spatial_id,'S6'],
            'backbones':spatial['config']['backbones'],
            'spatial_weight':weight,'preprocess':'face','balance':True,'augment':True,'weight_candidates':[.25,.5,.75],'selected_on':'validation'}
    atomic_json(out/'config.json',config)
    atomic_json(out/'summary.json',{'id':id,'config':config,'validation':calibration['before'],'calibration':calibration,
        'training_seconds':0,'training_cost_scope':f'reuses {spatial_id}/S6; validation weight selection only',
        'inference_ms_per_video':max(spatial['inference_ms_per_video'],temporal['inference_ms_per_video']),
        'latency_scope':'estimated shared-encoder upper branch cost; not independently benchmarked',
        'peak_vram_bytes':max(spatial['peak_vram_bytes'],temporal['peak_vram_bytes']),
        'checkpoint':str(out/'config.json'),'checkpoint_sha256':sha256(out/'config.json')})
    save_predictions(out/'validation_predictions.csv',rows,logits,calibration['temperature'])


def freeze():
    if (RUN/'freeze.json').exists():return
    summaries=[json.loads((RUN/c['id']/'summary.json').read_text(encoding='utf-8')) for c in CONFIGS]+[json.loads((RUN/id/'summary.json').read_text(encoding='utf-8')) for id in ['S7','S8']]
    # Prespecified deployment requirement: both class recalls > 0, then balanced accuracy,
    # macro-F1, AUC and cost. All test labels remain unused.
    eligible=[s for s in summaries if min(s['validation']['real_recall'],s['validation']['fake_recall'])>0]
    winner=max(eligible or summaries,key=lambda s:(s['validation']['balanced_accuracy'],s['validation']['macro_f1'],s['validation']['roc_auc'],-s['inference_ms_per_video']))
    record={'format':'unified_freeze_v1','manifest_sha256':sha256(RUN/'manifest.csv'),'selection_split':'validation',
            'selection_rule':'both recalls >0; highest balanced accuracy, macro-F1, ROC-AUC, lower measured cost',
            'champion':winner['id'],'experiments':{s['id']:{'checkpoint':s['checkpoint'],'checkpoint_sha256':s['checkpoint_sha256'],
                'temperature':s['calibration']['temperature'],'summary_sha256':sha256(RUN/s['id']/'summary.json')} for s in summaries},
            'optional_skipped':{'added_ECA_CBAM':'Deferred: prioritize frozen spatial, hybrid and temporal controlled comparisons',
                'temporal_transformer':'Deferred: small source-group cohort and laptop budget','full_backbone_finetuning':'Not run; all new encoders frozen for controlled feature-head transfer'}}
    atomic_json(RUN/'freeze.json',record);atomic_json(RUN/'champion.json',{'id':winner['id'],**record['experiments'][winner['id']],
        'selection_split':'validation','scope':'Local source-grouped DFDC subset, eight normalized video frames; image transfer unbenchmarked'})


def test_all(device):
    record=json.loads((RUN/'freeze.json').read_text(encoding='utf-8'))
    if record['manifest_sha256']!=sha256(RUN/'manifest.csv'):raise ValueError('Frozen manifest changed')
    rows=load_rows('test');y=np.array([int(r['label']=='FAKE') for r in rows])
    for id,contract in record['experiments'].items():
        out=RUN/id
        if (out/'test.json').exists():continue
        if sha256(contract['checkpoint'])!=contract['checkpoint_sha256'] or sha256(out/'summary.json')!=contract['summary_sha256']:raise ValueError('Frozen checkpoint/selection changed')
        # An interrupted test claim must be reviewed explicitly, never silently rerun.
        with (out/'test.claim').open('x') as claim:claim.write(json.dumps({'freeze_sha256':sha256(RUN/'freeze.json'),'checkpoint_sha256':contract['checkpoint_sha256']}))
        if id=='S0':
            legacy_predictions('test',device)
            logits=torch.load(out/'test_logits.pt',weights_only=True)
        elif id in {'S7','S8'}:
            config=json.loads((out/'config.json').read_text(encoding='utf-8'));w=config['spatial_weight']
            a=torch.load(RUN/config['components'][0]/'test_logits.pt',weights_only=True);b=torch.load(RUN/'S6/test_logits.pt',weights_only=True)
            p=w*probability(a)+(1-w)*probability(b);logits=torch.tensor(np.column_stack([1-p,p])).log()
        else:
            head,saved=read_checkpoint(out/'best.pt');x,_,_=feature_data(saved['config'],'test');x=(x-saved['feature_mean'])/saved['feature_std']
            with torch.inference_mode():logits=head.to(device)(x.to(device)).cpu()
        torch.save(logits,out/'test_logits.pt');p=probability(logits,contract['temperature'])
        atomic_json(out/'test.json',{'split':'held-out test','one_time_frozen_evaluation':True,
            'videos':len(rows),'source_groups':len({r['source_group'] for r in rows}),
            'raw':binary_metrics(y,probability(logits)),'calibrated':{**binary_metrics(y,p),'ece':ece(y,p)},
            'temperature_fit_split':'validation','checkpoint_sha256':contract['checkpoint_sha256']})
        save_predictions(out/'test_predictions.csv',rows,logits,contract['temperature'])
        print(f'test {id} bacc {binary_metrics(y,p)["balanced_accuracy"]:.4f}',flush=True)
    atomic_json(RUN/'all_fake_reference.json',{s:binary_metrics([int(r['label']=='FAKE') for r in load_rows(s)],np.ones(len(load_rows(s)))) for s in ['validation','test']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['features','train','freeze','test']);parser.add_argument('--split',choices=['train','validation','test'],default='train');args=parser.parse_args()
    device=setup()
    if args.stage=='features':extract(args.split,device)
    elif args.stage=='train':
        for config in CONFIGS[1:]:train_one(config,device)
        fusion()
        fusion('S8','S5')
    elif args.stage=='freeze':freeze()
    else:test_all(device)
