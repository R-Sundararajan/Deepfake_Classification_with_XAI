"""Validation-based ablations, champion selection and a single final image test."""
import argparse
import gc
import json
import time
from pathlib import Path

from tools.runtime import configure_runtime
configure_runtime()
import torch
from config import MODEL_OUTPUT_PATH
from evaluation.dfdc import load_baseline
from models.transfer import load_transfer_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget
from training.image_experiment import RUN_ROOT, ImageRows, evaluate, load_rows, plots, save_predictions, seed_all


def completed():
    results=[]
    for path in sorted(RUN_ROOT.glob('E*/summary.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('completed') and not row.get('smoke_test') and row.get('evaluation_split')=='validation': results.append(row)
    if not results: raise ValueError('No completed validation experiments')
    if len({r['manifest_sha256'] for r in results})!=1: raise ValueError('Experiments use different held-out data')
    for row in results:
        if sha256(row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    return results


def rank(rows):
    path=RUN_ROOT/'latency_benchmark.json'
    measured=json.loads(path.read_text(encoding='utf-8'))['experiments'] if path.exists() else {}
    return sorted(rows,key=lambda r:(round(r['validation']['roc_auc'],3),round(r['validation']['balanced_accuracy'],3),
                                    round(r['validation']['f1'],3),-measured.get(r['experiment_id'],{}).get('batch32_ms_per_image',r['validation']['inference_ms_per_image'])),reverse=True)


def benchmark():
    from PIL import Image
    from training.dfdc_cache import image_transform
    from training.image_experiment import synchronize
    path=RUN_ROOT/'latency_benchmark.json'
    if path.exists(): raise FileExistsError('Benchmark already exists')
    seed_all();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows,_=load_rows(RUN_ROOT/'dataset/manifest.csv')
    image_path=next(r['path'] for r in rows if r['split']=='validation')
    with Image.open(image_path) as image: tensor=image_transform()(image.convert('RGB')).unsqueeze(0).to(device)
    result={}
    for row in completed():
        model=load_baseline(row['checkpoint'],device) if row['experiment_id']=='E00' else load_transfer_checkpoint(row['checkpoint'],device)[0]
        timing={}
        with torch.inference_mode():
            for batch in [1,32]:
                inputs=tensor.repeat(batch,1,1,1)
                for _ in range(5): model(inputs).softmax(1)
                synchronize(device);observations=[]
                for _ in range(30):
                    start=time.perf_counter();model(inputs).softmax(1);synchronize(device)
                    observations.append((time.perf_counter()-start)*1000/batch)
                timing[f'batch{batch}_ms_per_image']=sum(observations)/len(observations)
                timing[f'batch{batch}_min_ms_per_image']=min(observations)
        result[row['experiment_id']]=timing
        del model,inputs;gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    report={'device':str(device),'experiments':result,'warmup_iterations':5,'timed_iterations':30,'precision':'FP32',
            'scope':'Model forward + softmax + synchronization, GPU-resident tensors; excludes transfer, image loading, face detection and Grad-CAM. One validation image repeated for controlled microbenchmark; no label or test-set evaluation.'}
    atomic_json(path,report);return report


def ablations():
    rows=[r for r in completed() if r['experiment_id'].startswith(('E01','E02'))]
    if len(rows)!=2: raise ValueError('E01 and E02 must finish before ablations')
    best=rank(rows)[0]
    for identifier,attention,augmentation in [('E03_attention','eca','basic'),('E04_jpeg','none','jpeg')]:
        config={**best['config'],'experiment_id':identifier,'model':{**best['config']['model'],'attention':attention},'augmentation':augmentation,
                'reference_experiment':best['experiment_id'],
                'justification':'Add lightweight ECA channel attention' if attention=='eca' else 'Train-time JPEG quality 50..95 with probability 0.5 models lossy media compression'}
        path=ROOT/'configs'/('execution_'+identifier+'.json')
        if path.exists(): raise FileExistsError('Ablation config already exists')
        atomic_json(path,config)
    return {'reference':best['experiment_id'],'configurations':['E03_attention','E04_jpeg']}


def finalize():
    rows=completed();best=rank(rows)[0];champion=ROOT/'results/champion.json'
    if champion.exists(): raise FileExistsError('Existing champion selection preserved')
    record={'format':'image_champion_v1','dataset':'Real vs Fake Faces','experiment_id':best['experiment_id'],'checkpoint':best['checkpoint'],
            'checkpoint_sha256':best['checkpoint_sha256'],'manifest_sha256':best['manifest_sha256'],
            'status':('Interim champion: existing ResNet-18' if best['experiment_id']=='E00' else 'Selected '+best['experiment_id'])+' trained on Real vs Fake Faces. Video inference is experimental; video accuracy has not been measured.',
            'selection_rule':'Validation image ROC-AUC, balanced accuracy, F1 rounded to 0.001; ties use lower common batch-32 inference latency',
            'candidate_ids':[r['experiment_id'] for r in rank(rows)],'validation':best['validation'],
            'limitations':'Legacy E00 used 100,000 training images; new models used the same 12,000-image subset. Image benchmark has no identity/source metadata. No labeled video metrics.'}
    atomic_json(champion,record)
    seed_all();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=RUN_ROOT/'champion_test';output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'test_started.json',{'champion_sha256':sha256(champion),'experiment_id':best['experiment_id']})
    if best['experiment_id']=='E00':
        # Milestone A already evaluated this exact preserved checkpoint on all
        # 20,000 test images. Reuse it, avoiding another test pass for selection.
        metrics=json.loads((RUN_ROOT/'E00_test/metrics.json').read_text(encoding='utf-8'))
        final={'experiment_id':'E00','dataset':'Real vs Fake Faces','evaluation_unit':'image','evaluation_split':'test',
               'metrics':metrics,'reused_milestone_A_test':True,'checkpoint_sha256':best['checkpoint_sha256'],
               'notes':'Selection used validation only. The test result was measured once during Milestone A and was not used in selection.'}
    else:
        rows_manifest,info=load_rows(RUN_ROOT/'dataset/manifest.csv')
        model,_=load_transfer_checkpoint(best['checkpoint'],device)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        metrics,predictions=evaluate(model,ImageRows(rows_manifest,'test'),device,64)
        save_predictions(output/'test_predictions.csv',predictions);plots(output,metrics)
        final={'experiment_id':best['experiment_id'],'dataset':'Real vs Fake Faces','evaluation_unit':'image','evaluation_split':'test',
               'metrics':metrics,'checkpoint_sha256':best['checkpoint_sha256'],
               'gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None}
    atomic_json(output/'summary.json',final)
    latency=json.loads((RUN_ROOT/'latency_benchmark.json').read_text(encoding='utf-8'))
    atomic_json(RUN_ROOT/'comparison.json',{'experiments':completed(),'champion':record,'final_test':final,'common_latency_benchmark':latency,'storage':check_budget()})
    return {'champion':record,'final_test':final}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['ablations','benchmark','finalize'])
    args=parser.parse_args();print(json.dumps(ablations() if args.action=='ablations' else benchmark() if args.action=='benchmark' else finalize(),indent=2))
