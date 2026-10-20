"""DFDC metadata, source-group splits and bounded streaming video sampling.

Metadata grouping prevents original-video leakage, not actor/identity leakage:
DFDC sample metadata does not provide reliable actor identity identifiers.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from tools.acquire_dfdc import PREFIX, sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget, owned_path


def safe_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*\.mp4',value) or '..' in value:
        raise ValueError('Invalid/unsafe video filename in metadata')
    return value


def load_metadata(path):
    with Path(path).open(encoding='utf-8') as f:
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d:raise ValueError('Duplicate metadata key')
                d[k]=v
            return d
        data=json.load(f,object_pairs_hook=unique)
    if not isinstance(data,dict) or not data:raise ValueError('Metadata must be a nonempty mapping')
    folded=set()
    for name,info in data.items():
        safe_name(name)
        if name.casefold() in folded:raise ValueError('Case-insensitive filename collision')
        folded.add(name.casefold())
        if not isinstance(info,dict) or info.get('label') not in {'REAL','FAKE'}:raise ValueError('Unknown/missing REAL/FAKE label')
        if info.get('original'):
            safe_name(info['original'])
        if info['label']=='FAKE' and not info.get('original'):
            raise ValueError(f'FAKE source relationship missing for {name}; leakage-safe splits unavailable')
    return data


def source_groups(metadata):
    parent={}
    def find(x):
        x=x.casefold()  # Match Windows paths and missing-original aliases.
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        aa,bb=find(a),find(b)
        parent[max(aa,bb)]=min(aa,bb)
    for name,info in metadata.items():
        find(name)
        if info.get('original'):union(name,info['original'])
        # Optional identity metadata, if an official release supplies it. The
        # competition sample normally has no such IDs; never infer identities.
        identities=[]
        for key in ['actor_ids','identity_ids','actor_id','person_id']:
            value=info.get(key)
            identities.extend(value if isinstance(value,list) else [value] if value is not None else [])
        for identity in identities:
            if not isinstance(identity,(str,int)):raise ValueError('Invalid optional identity identifier')
            union(name,'__identity_'+hashlib.sha256(str(identity).casefold().encode()).hexdigest())
    groups=defaultdict(list)
    for name in sorted(metadata):groups[find(name)].append(name)
    # Group IDs also encode missing external source names via the union root.
    return dict(groups)


def split_groups(metadata,seed=42,ratios=(.70,.15,.15)):
    if len(ratios)!=3 or any(r<=0 for r in ratios) or not math.isclose(sum(ratios),1):raise ValueError('Three positive split ratios must sum to one')
    groups=source_groups(metadata)
    if len(groups)<3:raise ValueError('Fewer than three independent source groups; cannot create three nonempty splits')
    names=['train','validation','test']
    total=Counter(i['label'] for i in metadata.values())
    best=None
    # Deterministic candidate search balances both labels while keeping whole
    # components intact; no frame/video from a component crosses partitions.
    for attempt in range(256):
        rng=random.Random(seed+attempt)
        ordered=list(sorted(groups));rng.shuffle(ordered)
        assignments={};counts=[Counter() for _ in names];sizes=[0]*3
        for g in ordered:
            gc=Counter(metadata[n]['label'] for n in groups[g])
            costs=[]
            for k in range(3):
                cost=0
                for j in range(3):
                    for label in ['REAL','FAKE']:
                        target=total[label]*ratios[j]
                        value=counts[j][label]+(gc[label] if j==k else 0)
                        cost+=(value-target)**2/max(target,1)
                costs.append(cost)
            choice=min(range(3),key=lambda k:(costs[k],sizes[k],k))
            assignments[g]=names[choice];counts[choice].update(gc);sizes[choice]+=len(groups[g])
        missing=sum(counts[j][label]==0 for j in range(3) for label in ['REAL','FAKE'] if total[label])
        empty=sum(x==0 for x in sizes)
        score=missing*1e6+empty*1e7+sum((counts[j][label]-total[label]*ratios[j])**2/max(total[label]*ratios[j],1) for j in range(3) for label in ['REAL','FAKE'])
        if best is None or score<best[0]:best=(score,assignments,counts)
    _,assignments,counts=best
    if any(not c for c in counts):raise ValueError('Could not produce three nonempty source-safe partitions')
    per_video={n:(g,assignments[g]) for g,ns in groups.items() for n in ns}
    warnings=[]
    if any(any(c[label]==0 for label in ['REAL','FAKE']) for c in counts):warnings.append('At least one split has a single class; AUC may be undefined. Source groups were not broken to force balance.')
    known_names={n.casefold() for n in metadata}
    external=sorted({v['original'].casefold() for v in metadata.values() if v.get('original') and v['original'].casefold() not in known_names})
    if external:warnings.append(f'{len(external)} original source names absent from sample; siblings remain grouped through these external names.')
    warnings.append('Source-video grouping does not guarantee complete actor disjointness. Optional official identity IDs are grouped when provided; sample metadata normally lacks them.')
    return per_video,dict(groups),warnings


def sample_indices(frame_count,fps,interval=1.0,max_frames=16):
    if frame_count<=0 or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid video frame count/fps')
    if not math.isfinite(interval) or interval<=0 or max_frames<1 or max_frames>512:raise ValueError('interval must be positive and max_frames in 1..512')
    # Compute only the retained index positions, even for long videos.
    step=max(1,round(fps*interval));n=(frame_count-1)//step+1
    positions=range(n) if n<=max_frames else [round(i*(n-1)/(max_frames-1)) for i in range(max_frames)] if max_frames>1 else [0]
    return sorted({p*step for p in positions})


def iter_frames(path,interval=1.0,max_frames=16):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Video cannot be opened')
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        for index in sample_indices(count,fps,interval,max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok or frame is None or not frame.size:raise ValueError(f'Frame decode failed at index {index}')
            yield index,index/fps,frame
    finally:cap.release()


def probe_video(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Video cannot be opened')
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        if n<1 or fps<=0 or not math.isfinite(fps):raise ValueError('Invalid frame metadata')
        for index in sorted({0,n//2,n-1}):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok or frame is None or not frame.size:raise ValueError('Probe decode failure')
        return {'fps':fps,'frame_count':n,'width':frame.shape[1],'height':frame.shape[0]}
    finally:cap.release()


class FaceCropper:
    def __init__(self,mode='full_frame',margin=1.3):
        import cv2
        if mode not in {'haar','full_frame'}:raise ValueError('Unknown face preprocessing')
        self.mode=mode;self.margin=margin
        self.cascade=None
        if mode=='haar':
            if not hasattr(cv2,'CascadeClassifier') or not hasattr(cv2,'data'):
                raise RuntimeError('This OpenCV build lacks Haar cascade support. Use --faces full_frame, or a separate compatible OpenCV build; the existing environment is preserved.')
            self.cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            if self.cascade.empty():raise RuntimeError('Bundled OpenCV Haar cascade unavailable')

    def __call__(self,frame):
        import cv2
        if self.cascade is None:return frame,'full_frame'
        # Bound detection compute on high-resolution videos; crop from original.
        h,w=frame.shape[:2];scale=min(1,640/max(h,w))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if scale<1:gray=cv2.resize(gray,None,fx=scale,fy=scale)
        faces=self.cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(24,24))
        if len(faces)==0:return frame,'no_face_full_frame_fallback'
        # Stable largest-face tie break. This is not identity tracking/alignment.
        x,y,bw,bh=sorted(faces,key=lambda f:(-int(f[2])*int(f[3]),int(f[0]),int(f[1])))[0]
        x,y,bw,bh=[v/scale for v in (x,y,bw,bh)]
        side=max(bw,bh)*self.margin;cx=x+bw/2;cy=y+bh/2
        x1=max(0,int(cx-side/2));x2=min(w,int(math.ceil(cx+side/2)))
        y1=max(0,int(cy-side/2));y2=min(h,int(math.ceil(cy+side/2)))
        return frame[y1:y2,x1:x2],'haar_largest_face'


def prepare(metadata_path,output,seed=42):
    metadata_path=Path(metadata_path).resolve();output=owned_path(output)
    if output.exists():raise FileExistsError('Prepared output exists; choose a new directory to preserve the manifest')
    metadata=load_metadata(metadata_path);mapping,groups,warnings=split_groups(metadata,seed)
    video_dir=metadata_path.parent;records=[];counts=defaultdict(Counter)
    provenance_path=video_dir.parent/'provenance.json'
    provenance=json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.exists() else {}
    official={r['name']:r for r in provenance.get('files',[])}
    verified=provenance.get('source')=='official_kaggle_dfdc' and provenance.get('complete') is True
    md=official.get(PREFIX+'metadata.json',{})
    verified=verified and md.get('sha256')==sha256(metadata_path)
    for name,info in sorted(metadata.items()):
        group,split=mapping[name];p=(video_dir/name).resolve()
        record={'filename':name,'label':info['label'],'original':info.get('original',''),'source_group':group,'split':split,'official_metadata_split':info.get('split',''),'path':str(p)}
        try:
            if not p.is_relative_to(video_dir):raise ValueError('Video path resolves outside sample directory')
            if not p.is_file():raise ValueError('Missing video')
            record.update({'bytes':p.stat().st_size,'sha256':sha256(p)})
            expected=official.get(PREFIX+name,{})
            if expected.get('sha256')!=record['sha256'] or expected.get('bytes')!=record['bytes']:verified=False
            record.update(probe_video(p));record['status']='valid'
        except (ValueError,OSError) as e:
            record['status']='invalid';record['error']=str(e);verified=False
        records.append(record);counts[split][record['label']+'_'+record['status']]+=1
    fields=['filename','path','label','original','source_group','split','official_metadata_split','status','bytes','sha256','fps','frame_count','width','height','error']
    check_budget(len(records)*4096+1024*1024);output.mkdir(parents=True)
    manifest=output/'manifest.csv'
    with manifest.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    report={'metadata_sha256':sha256(metadata_path),'manifest_sha256':sha256(manifest),'seed':seed,'split_ratios':[.70,.15,.15],
            'videos':len(records),'source_groups':len(groups),'label_counts':dict(Counter(v['label'] for v in metadata.values())),
            'split_counts':{k:dict(v) for k,v in counts.items()},'source_groups_by_split':dict(Counter(s for _,s in set(mapping.values()))),
            'invalid_videos':[{'filename':r['filename'],'error':r.get('error')} for r in records if r['status']!='valid'],
            'official_provenance_verified':verified,'warnings':warnings,'storage':check_budget()}
    atomic_json(output/'preparation.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,default=ROOT/'data/dfdc/train_sample_videos/metadata.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/dfdc/prepared_seed42')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();print(json.dumps(prepare(a.metadata,a.output,a.seed),indent=2))
