"""Validation-only model comparison and an explicit, single-use final holdout gate."""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.dfdc import binary_metrics, load_baseline, mean_fake_probability
from models.transfer import load_transfer_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget, owned_path
from training.dfdc_cache import FrameDataset, load_cache


def sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def aggregate_frames(frames):
    groups=defaultdict(list)
    for row in frames: groups[row['filename']].append(row)
    videos=[]
    for filename,rows in sorted(groups.items()):
        if len({r['label'] for r in rows})!=1 or len({r['source_group'] for r in rows})!=1:
            raise ValueError('Inconsistent labels/source group for a video')
        if len({r['frame_index'] for r in rows})!=len(rows): raise ValueError('Duplicate frame prediction')
        p=mean_fake_probability([r['fake_probability'] for r in rows])
        videos.append({'filename':filename,'source_group':rows[0]['source_group'],'label':rows[0]['label'],
                       'frames':len(rows),'fake_probability':p,'predicted_label':'FAKE' if p>=.5 else 'REAL'})
    return videos


def predict_dataset(model,dataset,device,batch_size=8):
    if not len(dataset): raise ValueError('Selected split contains no frames')
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval(); frames=[]; infer_seconds=0.; wall=time.perf_counter()
    # An untimed warm-up prevents CUDA initialization skew. No labels used.
    with torch.inference_mode():
        warm=dataset[0][0].unsqueeze(0).to(device); model(warm); sync(device)
        for inputs,targets,indices in loader:
            sync(device); start=time.perf_counter()
            probabilities=model(inputs.to(device)).float().softmax(1)[:,0].cpu().numpy(); sync(device)
            infer_seconds+=time.perf_counter()-start
            for index,probability in zip(indices.tolist(),probabilities):
                row=dataset.rows[index]
                frames.append({'filename':row['filename'],'source_group':row['source_group'],'label':row['label'],
                               'frame_index':int(row['frame_index']),'seconds':float(row['seconds']),'fake_probability':float(probability)})
    videos=aggregate_frames(frames)
    return {'video':binary_metrics([int(r['label']=='FAKE') for r in videos],[r['fake_probability'] for r in videos]),
            'frame':binary_metrics([int(r['label']=='FAKE') for r in frames],[r['fake_probability'] for r in frames]),
            'videos':videos,'frames':frames,'timing':{'model_batch_seconds':infer_seconds,'cached_eval_wall_seconds':time.perf_counter()-wall,
             'ms_per_frame':1000*infer_seconds/len(frames),'ms_per_video':1000*infer_seconds/len(videos),'frames_per_second':len(frames)/infer_seconds,
             'batch_size':batch_size,'scope':'cached RGB crops to model probabilities; excludes video decoding, detection, disk-cache creation and XAI'}}


def save_predictions(output,predictions,prefix='validation'):
    output=Path(output)
    for unit in ['video','frame']:
        rows=predictions[unit+'s']; path=output/f'{prefix}_{unit}_predictions.csv'
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {unit:sha256(output/f'{prefix}_{unit}_predictions.csv') for unit in ['video','frame']}


def source_bootstrap(videos,seed=42,draws=500):
    groups=defaultdict(list)
    for row in videos: groups[row['source_group']].append(row)
    keys=sorted(groups); rng=np.random.default_rng(seed); values=defaultdict(list)
    for _ in range(draws):
        sample=[row for key in rng.choice(keys,len(keys),replace=True) for row in groups[key]]
        metrics=binary_metrics([int(r['label']=='FAKE') for r in sample],[r['fake_probability'] for r in sample])
        for metric in ['accuracy','balanced_accuracy','f1','roc_auc']:
            if metrics[metric] is not None: values[metric].append(metrics[metric])
    return {'method':'source-group percentile bootstrap; conditional on this fixed sample and trained model','seed':seed,'draws':draws,
            'metrics':{k:{'lower':float(np.percentile(v,2.5)),'upper':float(np.percentile(v,97.5)),'defined_draws':len(v)} for k,v in values.items()}}


def select_champion(runs=ROOT/'results/experiments',output=ROOT/'results/champion.json'):
    output=owned_path(output)
    if output.exists(): raise FileExistsError('Champion selection already sealed; preserve it and the untouched final test record')
    candidates=[]; protocol=None
    for file in sorted(Path(runs).glob('*/summary.json')):
        row=json.loads(file.read_text(encoding='utf-8'))
        if not row.get('completed') or row.get('kind')!='transfer_training' or row.get('evaluation_split')!='validation': continue
        if row.get('dataset_provenance')!='official_kaggle_dfdc': continue
        key=(row['manifest_sha256'],json.dumps(row['preprocessing'],sort_keys=True))
        if protocol is not None and key!=protocol: raise ValueError('Candidate protocols differ; choose a comparable run directory')
        protocol=key
        if sha256(row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Candidate checkpoint hash mismatch')
        for unit,digest in row['prediction_sha256'].items():
            if sha256(file.parent/f'validation_{unit}_predictions.csv')!=digest: raise ValueError('Validation predictions changed')
        m=row['validation']['video']
        if m['roc_auc'] is not None and m['balanced_accuracy'] is not None: candidates.append(row)
    if not candidates: raise ValueError('No completed comparable official-data training experiments; no champion can be declared')
    # Metrics rounded only for tie handling: <=0.001 AUC/BAcc/F1 uses latency.
    ranked=sorted(candidates,key=lambda r:(round(r['validation']['video']['roc_auc'],3),round(r['validation']['video']['balanced_accuracy'],3),
                  round(r['validation']['video']['f1'],3),-r['validation']['timing']['ms_per_frame']),reverse=True)
    best=ranked[0]
    record={'format':'dfdc_champion_v1','experiment_id':best['experiment_id'],'checkpoint':best['checkpoint'],'checkpoint_sha256':best['checkpoint_sha256'],
            'cache':best['cache'],'manifest_sha256':best['manifest_sha256'],'preprocessing':best['preprocessing'],
            'selection_rule':'validation video ROC-AUC, balanced accuracy, F1 rounded to 0.001, then lower measured cached inference latency',
            'candidate_ids':[r['experiment_id'] for r in ranked],'validation':best['validation'], 'test_evaluated':False,
            'confidence':'Uncalibrated model probabilities; no forensic-certainty interpretation'}
    atomic_json(output,record); return record


def final_test(champion_path=ROOT/'results/champion.json',device='cuda'):
    champion_path=Path(champion_path); champion=json.loads(champion_path.read_text(encoding='utf-8'))
    output=owned_path(ROOT/'results/experiments/final_test')
    # The atomic claim is created before test predictions; a failed/interrupted
    # final run requires review and cannot silently permit repeated test tuning.
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'test_started.json',{'champion_sha256':sha256(champion_path),'experiment_id':champion['experiment_id']})
    if sha256(champion['checkpoint'])!=champion['checkpoint_sha256']: raise ValueError('Champion checkpoint changed')
    rows,cache=load_cache(champion['cache'])
    if cache['manifest_sha256']!=champion['manifest_sha256']: raise ValueError('Champion manifest changed')
    model,_=load_transfer_checkpoint(champion['checkpoint'],device)
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats()
    result=predict_dataset(model,FrameDataset(champion['cache'],rows,'test'),device)
    hashes=save_predictions(output,result,'test')
    summary={'completed':True,'evaluation_split':'test','dataset_provenance':'official_kaggle_dfdc','experiment_id':champion['experiment_id'],
             'checkpoint_sha256':champion['checkpoint_sha256'],'champion_sha256':sha256(champion_path),'manifest_sha256':cache['manifest_sha256'],
             'metrics':result['video'],'frame_metrics':result['frame'],'timing':result['timing'],'videos':len(result['videos']),
             'prediction_sha256':hashes,'source_bootstrap_95ci':source_bootstrap(result['videos']),
             'gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2 if torch.device(device).type=='cuda' else None}
    atomic_json(output/'summary.json',summary)
    make_plots(result['videos'],output)
    return summary


def make_plots(videos,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
    y=[int(r['label']=='FAKE') for r in videos]; p=[r['fake_probability'] for r in videos]
    for name,kind in [('confusion_matrix',ConfusionMatrixDisplay),('roc',RocCurveDisplay),('precision_recall',PrecisionRecallDisplay)]:
        if name!='confusion_matrix' and len(set(y))<2: continue
        if name=='confusion_matrix': kind.from_predictions(y,[int(x>=.5) for x in p],labels=[0,1],display_labels=['REAL','FAKE'],colorbar=False)
        else: kind.from_predictions(y,p)
        plt.tight_layout(); plt.savefig(Path(output)/(name+'.png'),dpi=160); plt.close()


def baseline(cache_path,run_id='legacy_resnet18_validation',device='cuda'):
    rows,cache=load_cache(cache_path)
    output=owned_path(ROOT/'results/experiments'/run_id); output.mkdir(parents=True,exist_ok=False)
    path=ROOT/'models/best_resnet18.pth'; before=sha256(path)
    model=load_baseline(path,device)
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats()
    result=predict_dataset(model,FrameDataset(cache_path,rows,'validation'),device)
    hashes=save_predictions(output,result)
    summary={'completed':True,'kind':'legacy_cross_dataset','evaluation_split':'validation','dataset_provenance':'official_kaggle_dfdc','experiment_id':run_id,
             'model':'Existing image-trained ResNet-18','manifest_sha256':cache['manifest_sha256'],'preprocessing':cache['preprocessing'],'cache':str(cache_path),
             'validation':{k:v for k,v in result.items() if k not in {'videos','frames'}},'checkpoint':str(path),'checkpoint_sha256':before,
             'prediction_sha256':hashes,'notes':'Cross-dataset validation only, before DFDC training. Held-out test not read. Not eligible as a newly trained champion.'}
    summary.update(parameters=sum(p.numel() for p in model.parameters()),checkpoint_bytes=path.stat().st_size,training_seconds=0,
                   gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2 if torch.device(device).type=='cuda' else None)
    if sha256(path)!=before: raise RuntimeError('Legacy checkpoint changed')
    atomic_json(output/'summary.json',summary); make_plots(result['videos'],output); return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['baseline','select','test'])
    parser.add_argument('--cache',type=Path,default=ROOT/'data/dfdc/cache_yunet_4')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    from tools.runtime import configure_runtime
    configure_runtime(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    result=baseline(args.cache,device=args.device) if args.action=='baseline' else select_champion() if args.action=='select' else final_test(device=args.device)
    print(json.dumps(result,indent=2))
