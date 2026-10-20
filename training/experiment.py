"""One config, one controlled transfer-learning run. No architecture queue."""
import argparse
import copy
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from tools.runtime import configure_runtime
configure_runtime()  # Before importing torch/CUDA or downloading trusted weights.

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from evaluation.experiments import make_plots, predict_dataset, save_predictions
from models.transfer import CLASS_TO_IDX, WEIGHTS, TransferClassifier, load_transfer_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget, owned_path
from tools.runtime import memory_snapshot
from training.dfdc_cache import FrameDataset, load_cache


DEFAULTS={'seed':42,'batch_size':4,'accumulation_steps':8,'head_epochs':3,'fine_tune_epochs':0,'fine_tune_blocks':1,
          'head_lr':.001,'backbone_lr':.00001,'weight_decay':.0001,'patience':2,'augmentation':'basic','class_weighted_loss':True,
          'amp':True,'input_size':224,'max_minutes':30,'device':'cuda','cache':'data/dfdc/cache_yunet_4'}


def read_config(path):
    config={**DEFAULTS,**json.loads(Path(path).read_text(encoding='utf-8'))}
    if not str(config.get('experiment_id','')).replace('_','').replace('-','').isalnum(): raise ValueError('Safe unique experiment_id required')
    if config['input_size']!=224 or not 1<=config['batch_size']<=64 or not 1<=config['accumulation_steps']<=32: raise ValueError('Invalid image/batch configuration')
    if config['head_epochs']<0 or config['fine_tune_epochs']<0 or config['head_epochs']+config['fine_tune_epochs']<1: raise ValueError('No training epochs')
    if config['fine_tune_epochs'] and not (config['head_epochs'] or config.get('initialize_from')): raise ValueError('Train a frozen head before selective fine-tuning')
    if config['max_minutes']<=0 or config['patience']<1: raise ValueError('Invalid time/patience budget')
    return config


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def save_torch(path,payload):
    path=owned_path(path); temporary=path.with_suffix('.pending.pt')
    if temporary.exists(): raise FileExistsError('Unfinished checkpoint preserved')
    estimated=sum(v.numel()*v.element_size() for v in payload['model_state_dict'].values())*4+1024**2
    check_budget(estimated); torch.save(payload,temporary); temporary.replace(path)


def train(config_path,resume=False,synthetic_test=False):
    config=read_config(config_path);seed_all(config['seed'])
    cache_path=(ROOT/config['cache']).resolve(); rows,cache=load_cache(cache_path,require_official=not synthetic_test)
    if synthetic_test and cache.get('official_provenance_verified'): raise ValueError('Synthetic test must use synthetic fixtures')
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if device.type=='cuda' and torch.cuda.mem_get_info()[0]<1024**3: raise RuntimeError('Less than 1 GiB GPU memory free')
    folder=ROOT/('results/smoke' if synthetic_test else 'results/experiments')/config['experiment_id']; output=owned_path(folder)
    if output.exists() and not resume: raise FileExistsError('Experiment exists; use --resume or a new ID')
    if resume and not (output/'last.pt').is_file(): raise FileNotFoundError('Resumable last.pt is missing')
    if (output/'summary.json').exists(): raise FileExistsError('Completed experiment is immutable; use a new ID')
    output.mkdir(parents=True,exist_ok=True)
    if resume:
        if json.loads((output/'config.json').read_text(encoding='utf-8'))!=config: raise ValueError('Resume config differs from recorded config')
    else: atomic_json(output/'config.json',config)
    datasets={split:FrameDataset(cache_path,rows,split,config['augmentation'],config['input_size']) for split in ['train','validation']}
    for name,dataset in datasets.items():
        if len({row['label'] for row in dataset.rows})!=2: raise ValueError(f'{name} requires both classes')
    spec=config['model']
    prior=None
    if config.get('initialize_from'):
        initial=(ROOT/config['initialize_from']).resolve()
        model,prior=load_transfer_checkpoint(initial,device)
        if model.spec!=dict(backbone=spec['backbone'],attention=spec.get('attention','none'),fusion=spec.get('fusion','none'),dropout=spec.get('dropout',.2)):
            raise ValueError('Initialization model differs from config')
        if prior['manifest_sha256']!=cache['manifest_sha256'] or prior['preprocessing']!=cache['preprocessing']:
            raise ValueError('Initialization checkpoint data/preprocessing differs')
    else:
        check_budget(256*1024**2)  # Pinned pretrained weights plus first checkpoints.
        model=TransferClassifier(**spec,pretrained=not synthetic_test and not resume).to(device)
    train_counts=Counter(int(r['target']) for r in datasets['train'].rows)
    weights=torch.tensor([sum(train_counts.values())/(2*train_counts[i]) for i in range(2)],device=device)
    criterion=nn.CrossEntropyLoss(weight=weights if config['class_weighted_loss'] else None)
    phases=[('head',0)]*config['head_epochs']+[('fine_tune',config['fine_tune_blocks'])]*config['fine_tune_epochs']
    start_epoch=0;best_score=(-1.,-1.,-1.);best_epoch=-1;bad_epochs=0;history=[];elapsed_prior=0.; optimizer=None;current_phase=None
    scaler=torch.amp.GradScaler('cuda',enabled=config['amp'] and device.type=='cuda')
    resumed=None
    if resume:
        resumed=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        if resumed['manifest_sha256']!=cache['manifest_sha256']: raise ValueError('Resume manifest changed')
        model.load_state_dict(resumed['model_state_dict']);start_epoch=resumed['next_epoch'];best_score=tuple(resumed['best_score']);best_epoch=resumed['best_epoch'];bad_epochs=resumed['bad_epochs'];history=resumed['history'];elapsed_prior=resumed['elapsed_seconds']
        torch.set_rng_state(resumed['torch_rng_state']);random.setstate(resumed['python_rng_state'])
        if device.type=='cuda' and resumed.get('cuda_rng_state') is not None: torch.cuda.set_rng_state(resumed['cuda_rng_state'])
        scaler.load_state_dict(resumed['scaler_state_dict'])
    start=time.perf_counter();peak_ram=memory_snapshot().get('process_peak_bytes',0)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()

    def payload(epoch):
        return {'format':'dfdc_transfer_v1','model_spec':model.spec,'model_state_dict':model.state_dict(),'class_to_idx':CLASS_TO_IDX,
                'fine_tune_blocks':model.fine_tune_blocks,'image_size':224,'manifest_sha256':cache['manifest_sha256'],'preprocessing':cache['preprocessing'],
                'experiment_id':config['experiment_id'],'pretrained_source':str(WEIGHTS[spec['backbone']]) if not synthetic_test else 'random initialization for synthetic software test only',
                'pretrained_url':WEIGHTS[spec['backbone']].url if not synthetic_test else None,
                'confidence':'uncalibrated','next_epoch':epoch+1,'best_score':list(best_score),'best_epoch':best_epoch,'bad_epochs':bad_epochs,
                'history':history,'elapsed_seconds':elapsed_prior+time.perf_counter()-start,'optimizer_state_dict':optimizer.state_dict(),
                'scaler_state_dict':scaler.state_dict(),'torch_rng_state':torch.get_rng_state(),'python_rng_state':random.getstate(),
                'cuda_rng_state':torch.cuda.get_rng_state() if device.type=='cuda' else None}

    for epoch in range(start_epoch,len(phases)):
        phase,blocks=phases[epoch]
        if phase!=current_phase:
            if current_phase=='head' and (output/'best.pt').exists(): model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model_state_dict'])
            model.set_trainable(blocks); current_phase=phase
            backbone=[p for p in model.features.parameters() if p.requires_grad]
            head=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('features.')]
            groups=[{'params':head,'lr':config['head_lr']}]
            if backbone: groups.append({'params':backbone,'lr':config['backbone_lr']})
            optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
            if resumed and epoch==start_epoch and resumed['phase']==phase: optimizer.load_state_dict(resumed['optimizer_state_dict'])
        generator=torch.Generator().manual_seed(config['seed']+epoch)
        loader=DataLoader(datasets['train'],batch_size=config['batch_size'],shuffle=True,num_workers=0,generator=generator)
        model.train(); optimizer.zero_grad(set_to_none=True);loss_sum=0.;count=0;epoch_start=time.perf_counter()
        for step,(inputs,targets,_) in enumerate(loader):
            if time.perf_counter()-start>config['max_minutes']*60:
                atomic_json(output/'interrupted.json',{'completed':False,'reason':'Configured time budget exceeded; restart from previous completed epoch using --resume','epoch':epoch,'resumable':(output/'last.pt').exists()})
                raise RuntimeError('Run time budget exceeded; no completed research result recorded')
            # The final partial accumulation group uses its actual number of steps.
            group_steps=min(config['accumulation_steps'],len(loader)-(step//config['accumulation_steps'])*config['accumulation_steps'])
            with torch.autocast(device_type=device.type,enabled=config['amp'] and device.type=='cuda'):
                logits=model(inputs.to(device)); loss=criterion(logits,targets.to(device)); scaled=loss/group_steps
            scaler.scale(scaled).backward();loss_sum+=float(loss.detach())*len(targets);count+=len(targets)
            if (step+1)%config['accumulation_steps']==0 or step+1==len(loader):
                scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if step%25==0: check_budget();peak_ram=max(peak_ram,memory_snapshot().get('process_peak_bytes',0))
        result=predict_dataset(model,datasets['validation'],device,config['batch_size'])
        metrics=result['video'];score=tuple(metrics[k] if metrics[k] is not None else -1. for k in ['roc_auc','balanced_accuracy','f1'])
        history.append({'epoch':epoch,'phase':phase,'training_loss':loss_sum/count,'validation_video':metrics,'epoch_seconds':time.perf_counter()-epoch_start})
        improved=score>best_score
        if improved: best_score=score;best_epoch=epoch;bad_epochs=0
        else: bad_epochs+=1
        saved=payload(epoch);saved['phase']=phase
        save_torch(output/'last.pt',saved)
        if improved:
            save_torch(output/'best.pt',saved);save_predictions(output,result)
        print(json.dumps({'experiment_id':config['experiment_id'],'epoch':epoch,'phase':phase,'train_loss':history[-1]['training_loss'],'validation_video_auc':metrics['roc_auc'],'balanced_accuracy':metrics['balanced_accuracy']}),flush=True)
        if bad_epochs>=config['patience']: break
    if not (output/'best.pt').exists(): raise RuntimeError('No completed epoch/checkpoint; no metrics published')
    model,saved=load_transfer_checkpoint(output/'best.pt',device)
    result=predict_dataset(model,datasets['validation'],device,config['batch_size']);hashes=save_predictions(output,result)
    summary={'completed':True,'kind':'synthetic_smoke' if synthetic_test else 'transfer_training','dataset_provenance':'synthetic_fixture' if synthetic_test else 'official_kaggle_dfdc',
             'evaluation_split':'validation','experiment_id':config['experiment_id'],'model':model.spec,'pretrained_source':saved['pretrained_source'],
             'manifest_sha256':cache['manifest_sha256'],'cache':str(cache_path),'preprocessing':cache['preprocessing'],'config':config,
             'training_frames':len(datasets['train']),'training_videos':len({r['filename'] for r in datasets['train'].rows}),
             'validation_frames':len(datasets['validation']),'validation_videos':len(result['videos']),'best_epoch':best_epoch,
             'training_seconds':elapsed_prior+time.perf_counter()-start,'gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
             'peak_process_ram_bytes':peak_ram,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'checkpoint':str(output/'best.pt'),'checkpoint_sha256':sha256(output/'best.pt'),'checkpoint_bytes':(output/'best.pt').stat().st_size,
             'prediction_sha256':hashes,'validation':{k:v for k,v in result.items() if k not in {'videos','frames'}},'history':history,
             'versions':{'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'numpy':np.__version__,'python':sys.version},'pretrained_url':saved.get('pretrained_url'),'storage':check_budget(),
             'limitations':'Official training sample only. Source disjointness does not imply actor disjointness. Uncalibrated probabilities. Cached-crop latency excludes decoding/detection/XAI.'}
    atomic_json(output/'summary.json',summary);make_plots(result['videos'],output)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try: train(args.config,args.resume)
    except FileNotFoundError as error: raise SystemExit(f'Missing input: {error}. Acquire, prepare and cache the official DFDC sample first.')
