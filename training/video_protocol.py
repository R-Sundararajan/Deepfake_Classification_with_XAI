"""Frozen local DFDC protocol: source groups, eight normalized timestamps, bounded cache.

Only existing local media is inspected. Missing labels are logically quarantined.
No claim of official distribution provenance or actor disjointness is made.
"""
import argparse
import csv
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget
from training.dfdc_data import load_metadata, split_groups
from training.dfdc_cache import make_cropper

RUN = ROOT / 'results/dfdc_unified_20261019'
CACHE = ROOT / 'data/dfdc/runtime/unified_frames'
SPEC = {'size':224, 'frames_per_video':8, 'sampling':'normalized_bin_centers',
        'faces':'yunet','align':True,'margin':1.3,'normalization':'ImageNet',
        'no_face_policy':'full_frame_fallback'}
_local=threading.local()


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def normalized_indices(count, frames=8):
    if count < frames: raise ValueError('Fewer than eight distinct frames; reject rather than pad/replicate')
    return [min(count-1,int((i+.5)*count/frames)) for i in range(frames)]


def normalized_frames(path):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise ValueError('Video cannot be opened')
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps<=0: raise ValueError('Invalid FPS')
        indices=normalized_indices(count)
        if count<=512:
            # For short GOP-compressed clips, eight random seeks repeatedly
            # decode the same GOP. Grab once in order and retrieve only the
            # identical selected positions; no extra frame becomes model input.
            selected=set(indices)
            for index in range(indices[-1]+1):
                if not cap.grab():raise ValueError(f'Decode failure at frame {index}')
                if index in selected:
                    ok,frame=cap.retrieve()
                    if not ok or frame is None:raise ValueError(f'Decode failure at frame {index}')
                    yield index,index/fps,frame
        else:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok or frame is None: raise ValueError(f'Decode failure at frame {index}')
                yield index,index/fps,frame
    finally: cap.release()


def statistics(values):
    x=np.asarray(values,float)
    if not len(x):return {}
    return {'count':len(x),'total_seconds':float(x.sum()),'minimum':float(x.min()),
            'p10':float(np.percentile(x,10)),'p25':float(np.percentile(x,25)),
            'median':float(np.median(x)),'mean':float(x.mean()),'p75':float(np.percentile(x,75)),
            'p90':float(np.percentile(x,90)),'maximum':float(x.max())}


def audit():
    if (RUN/'audit.json').exists():
        report=json.loads((RUN/'audit.json').read_text(encoding='utf-8'));
        if sha256(RUN/'manifest.csv')!=report['manifest_sha256']: raise ValueError('Frozen manifest changed')
        return report
    RUN.mkdir(parents=True,exist_ok=True)
    md=ROOT/'data/dfdc/metadata/metadata.json'; metadata=load_metadata(md)
    paths=sorted((ROOT/'data/dfdc/train_sample_videos').glob('*.mp4'))
    if len({p.name.casefold() for p in paths})!=len(paths):raise ValueError('Duplicate filenames')
    present={p.name for p in paths}; labeled={n:v for n,v in metadata.items() if n in present}
    mapping,groups,warnings=split_groups(labeled,42)
    quarantined=[{'filename':p.name,'path':str(p),'sha256':sha256(p),'reason':'no metadata label; excluded from all supervised stages'} for p in paths if p.name not in labeled]
    atomic_json(RUN/'quarantine.json',quarantined)
    rows=[]; started=time.perf_counter()
    for i,p in enumerate(paths,1):
        if p.name not in labeled:continue
        info=labeled[p.name];group,split=mapping[p.name]
        cap=cv2.VideoCapture(str(p));error='';count=fps=width=height=0
        try:
            if not cap.isOpened():raise ValueError('cannot open')
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not np.isfinite(fps) or fps<=0:raise ValueError('invalid fps')
            # Decode every timestamp that will enter the common protocol.
            for index in normalized_indices(count):
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok or frame is None:raise ValueError(f'decode failure at {index}')
        except (ValueError,cv2.error) as exc:error=str(exc)
        finally:cap.release()
        rows.append({'filename':p.name,'path':str(p.resolve()),'label':info['label'],
                     'original':info.get('original',''),'source_group':group,'split':split,
                     'sha256':sha256(p),'bytes':p.stat().st_size,'frame_count':count,'fps':fps,
                     'width':width,'height':height,'duration_seconds':count/fps if fps>0 else 0,
                     'status':'valid' if not error else 'excluded_decode_error','error':error})
        if i%50==0:print(f'audit {i}/{len(paths)}',flush=True)
    # Exact duplicates must also not cross splits.
    seen={}
    for row in rows:
        if seen.setdefault(row['sha256'],row['split'])!=row['split']:raise ValueError('Exact duplicate across splits; regroup before training')
    write_csv(RUN/'manifest.csv',rows)
    counts=Counter(r['label'] for r in rows)
    report={'format':'local_dfdc_normalized8_v1','metadata_path':str(md),'metadata_sha256':sha256(md),
            'local_videos':len(paths),'labeled_videos':len(rows),'label_counts':dict(counts),
            'unlabeled_quarantined':quarantined,'conflicting_labels':0,'metadata_entries':len(metadata),
            'provenance':'Existing local subset matched by filename to supplied Kaggle community-mirror metadata; not certified official distribution',
            'manifest_sha256':sha256(RUN/'manifest.csv'),'seed':42,'source_groups':len(groups),
            'split_counts':{s:{'videos':sum(r['split']==s for r in rows),
                'labels':dict(Counter(r['label'] for r in rows if r['split']==s)),
                'source_groups':len({r['source_group'] for r in rows if r['split']==s})} for s in ['train','validation','test']},
            'duration_by_class':{c:statistics([r['duration_seconds'] for r in rows if r['label']==c and r['status']=='valid']) for c in ['REAL','FAKE']},
            'decode_errors':[r for r in rows if r['error']],
            'all_fake_accuracy':counts['FAKE']/len(rows),'warnings':warnings,
            'probe_scope':'All labeled containers and eight normalized sampled frames per video; not exhaustive all-frame decoding',
            'preprocessing':SPEC,'audit_seconds':time.perf_counter()-started,'storage':check_budget()}
    atomic_json(RUN/'audit.json',report)
    plots(rows)
    return report


def plots(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=RUN/'figures';folder.mkdir(exist_ok=True)
    values=[[r['duration_seconds'] for r in rows if r['label']==c and r['status']=='valid'] for c in ['REAL','FAKE']]
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot(values,tick_labels=['REAL','FAKE'],showmeans=True)
    ax.set_ylim(0,max(max(v) for v in values)*1.15)
    if max(max(v) for v in values)-min(min(v) for v in values)<.01:
        ax.text(.5,.92,'Both classes ≈10.01 s; sub-millisecond estimation differences',ha='center',transform=ax.transAxes,fontsize=9)
    ax.set(ylabel='Duration (seconds)',title='Local DFDC duration by class');fig.tight_layout();fig.savefig(folder/'duration.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    bins=np.linspace(0,max(max(v) for v in values)*1.15,24)
    for c,v in zip(['REAL','FAKE'],values):axes[0].hist(v,bins=bins,alpha=.5,label=c,density=True)
    axes[0].set(xlabel='Seconds',ylabel='Density',title='Duration distribution');axes[0].legend()
    ss=['train','validation','test'];x=np.arange(3)
    for k,c in enumerate(['REAL','FAKE']):axes[1].bar(x+(k-.5)*.35,[sum(r['split']==s and r['label']==c for r in rows) for s in ss],.35,label=c)
    axes[1].set_xticks(x,ss);axes[1].set(ylabel='Videos',title='Source-grouped natural split');axes[1].legend()
    fig.tight_layout();fig.savefig(folder/'distribution.png',dpi=150);plt.close(fig)


def load_rows(split=None):
    report=json.loads((RUN/'audit.json').read_text(encoding='utf-8'))
    if sha256(RUN/'manifest.csv')!=report['manifest_sha256']:raise ValueError('Manifest hash mismatch')
    with (RUN/'manifest.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    return [r for r in rows if r['status']=='valid' and (split is None or r['split']==split)]


def _cache_one(row):
        dest=CACHE/(Path(row['filename']).stem+'.npz')
        if dest.exists():
            with np.load(dest,allow_pickle=False) as data:
                if str(data['source_sha256'])!=row['sha256'] or data['raw'].shape!=(8,224,224,3):raise ValueError('Stale cache')
                return data['methods'].tolist()
        if not hasattr(_local,'cropper'):_local.cropper=make_cropper(SPEC)
        if sha256(row['path'])!=row['sha256']:raise ValueError('Video changed')
        raw=[];face=[];methods=[];indices=[];seconds=[]
        for index,second,bgr in normalized_frames(row['path']):
            cropped,method=_local.cropper(bgr)
            raw.append(np.asarray(Image.fromarray(bgr[:,:,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR)))
            face.append(np.asarray(Image.fromarray(cropped[:,:,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR)))
            methods.append(method);indices.append(index);seconds.append(second)
        temp=dest.with_suffix('.part')
        with temp.open('wb') as stream:np.savez(stream,raw=np.stack(raw),face=np.stack(face),methods=np.array(methods),indices=indices,seconds=seconds,source_sha256=row['sha256'])
        temp.replace(dest)
        return methods


def cache():
    audit();CACHE.mkdir(parents=True,exist_ok=True)
    rows=load_rows();counts=Counter();started=time.perf_counter()
    check_budget(len(rows)*8*224*224*3*2+1024**2)
    # Two bounded CPU workers only; each owns its detector/decoder. Model
    # experiments still run sequentially. Existing complete files are reused.
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i,methods in enumerate(executor.map(_cache_one,rows),1):
            counts.update(methods)
            if i%25==0:print(f'cache {i}/{len(rows)}',flush=True)
    atomic_json(RUN/'cache.json',{'complete':True,'videos':len(rows),'frames':len(rows)*8,'shape':[8,224,224,3],
        'format':'lossless uint8 npz, raw and aligned variants; no test labels used by preprocessing',
        'preprocessing':SPEC,'crop_methods':dict(counts),'seconds':time.perf_counter()-started,
        'timing_scope':'Final completed cache pass; reuses complete files from interrupted preparation passes',
        'CPU_workers':2,'storage':check_budget()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','cache']);args=parser.parse_args()
    from tools.runtime import configure_runtime
    configure_runtime();cv2.setNumThreads(2)
    print(json.dumps(audit() if args.stage=='audit' else cache(),indent=2))
