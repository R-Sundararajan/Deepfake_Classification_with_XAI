"""Reusable deterministic, bounded face crops with manifest/cache integrity checks."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD
from models.transfer import CLASS_TO_IDX
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget, owned_path, tree_bytes
from training.dfdc_data import FaceCropper, iter_frames


def load_manifest(path,require_official=True):
    path=Path(path).resolve()
    preparation=json.loads((path.parent/'preparation.json').read_text(encoding='utf-8'))
    if sha256(path)!=preparation['manifest_sha256']: raise ValueError('Manifest hash mismatch')
    if require_official and not preparation.get('official_provenance_verified'): raise ValueError('Verified official DFDC provenance required')
    with path.open(newline='',encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    seen=set(); groups={}; hashes={}
    for row in rows:
        if row['filename'].casefold() in seen: raise ValueError('Duplicate manifest filename')
        seen.add(row['filename'].casefold())
        if row['label'] not in {'REAL','FAKE'} or row['split'] not in {'train','validation','test'}: raise ValueError('Invalid manifest label/split')
        group=row['source_group']
        if not group: raise ValueError('Missing source group')
        if groups.setdefault(group,row['split'])!=row['split']: raise ValueError('Source-group leakage')
        if row.get('sha256') and hashes.setdefault(row['sha256'],row['split'])!=row['split']: raise ValueError('Exact duplicate video crosses splits')
        if row.get('original'):
            source=next((r for r in rows if r['filename'].casefold()==row['original'].casefold()),None)
            if source is not None and source['source_group']!=group: raise ValueError('Original/derivative group mismatch')
    if not rows: raise ValueError('Empty manifest')
    return rows,preparation


class YuNetCropper:
    """Largest detected face; optional landmark affine alignment, bounded detector size."""
    def __init__(self,model_path,align=True,margin=1.3,score_threshold=.9):
        model_path=Path(model_path)
        if not model_path.is_file(): raise FileNotFoundError('YuNet weights missing. Run python -m tools.fetch_yunet')
        if not hasattr(cv2,'FaceDetectorYN'): raise RuntimeError('OpenCV FaceDetectorYN unavailable')
        self.detector=cv2.FaceDetectorYN.create(str(model_path),'',(320,320),score_threshold,.3,5000)
        self.align=align; self.margin=margin

    def __call__(self,frame):
        h,w=frame.shape[:2]; scale=min(1.,640/max(h,w))
        self.last_metadata = {'original_size_wh': [w, h], 'detected': False, 'bounds_xyxy': [0, 0, w, h]}
        small=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))))
        self.detector.setInputSize((small.shape[1],small.shape[0])); _,faces=self.detector.detect(small)
        if faces is None or not len(faces): return frame,'no_face_full_frame_fallback'
        face=sorted(faces,key=lambda f:(-float(f[2]*f[3]),float(f[0]),float(f[1])))[0].copy()
        face[:14]/=scale
        x,y,bw,bh = [float(value) for value in face[:4]]
        self.last_metadata.update(detected=True, detector_bbox_xyxy=[x,y,x+bw,y+bh],
                                  landmarks_xy=face[4:14].reshape(5,2).tolist(), margin=self.margin)
        if self.align:
            points=face[4:10].reshape(3,2).astype(np.float32)
            target=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366]],np.float32)*2
            target=(target-112)/self.margin+112
            matrix=cv2.getAffineTransform(points,target)
            corners=np.array([[0,0,1],[223,0,1],[223,223,1],[0,223,1]],np.float64)
            self.last_metadata.update(affine_original_to_224=matrix.tolist(),
                                      input_corners_original_xy=(corners@cv2.invertAffineTransform(matrix).T).tolist())
            return cv2.warpAffine(frame,matrix,(224,224),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE),'yunet_landmark_aligned'
        x,y,bw,bh=face[:4]; side=max(bw,bh)*self.margin; cx=x+bw/2;cy=y+bh/2
        x1=max(0,int(cx-side/2));y1=max(0,int(cy-side/2));x2=min(w,math.ceil(cx+side/2));y2=min(h,math.ceil(cy+side/2))
        self.last_metadata['bounds_xyxy']=[x1,y1,x2,y2]
        return frame[y1:y2,x1:x2],'yunet_largest_face'


def make_cropper(spec):
    if spec['faces']=='yunet': return YuNetCropper(spec.get('detector_path',ROOT/'data/dfdc/runtime/yunet.onnx'),spec.get('align',True),spec.get('margin',1.3))
    if spec.get('align',False): raise ValueError('Landmark alignment requires YuNet')
    return FaceCropper(spec['faces'],spec.get('margin',1.3))


def prepare_cache(manifest,output,frames=4,interval=1.,faces='yunet',align=True,margin=1.3,require_official=True):
    manifest=Path(manifest).resolve(); output=owned_path(output)
    rows,preparation=load_manifest(manifest,require_official)
    spec={'size':224,'frames_per_video':frames,'interval':interval,'faces':faces,'align':align,'margin':margin,
          'no_face_policy':'full_frame_fallback','color':'RGB','normalization':'ImageNet','sampling':'uniformly retain positions from interval grid'}
    if not 1<=frames<=32: raise ValueError('Cache supports 1..32 sampled frames per video')
    if output.exists(): raise FileExistsError('Cache directory exists; verify/reuse it or choose a new output')
    check_budget(len(rows)*frames*224*224*3+16*1024**2)
    cropper=make_cropper(spec); output.mkdir(parents=True)
    records=[]; errors=[]; methods=Counter()
    for video_index,row in enumerate(rows,1):
        local=[]
        try:
            if row['status']!='valid': raise ValueError('Invalid video in manifest')
            if sha256(row['path'])!=row['sha256']: raise ValueError('Video changed since manifest creation')
            for index,seconds,bgr in iter_frames(row['path'],interval,frames):
                crop,method=cropper(bgr)
                rgb=Image.fromarray(crop[:,:,::-1].copy()).resize((224,224),Image.Resampling.BILINEAR)
                relative=Path('frames')/row['split']/f"{Path(row['filename']).stem}_{index:08d}.png"
                dest=output/relative; dest.parent.mkdir(parents=True,exist_ok=True)
                check_budget(256*1024); rgb.save(dest,compress_level=3)
                local.append({'filename':row['filename'],'source_group':row['source_group'],'split':row['split'],'label':row['label'],'target':CLASS_TO_IDX[row['label'].lower()],
                              'frame_index':index,'seconds':seconds,'relative_path':relative.as_posix(),'sha256':sha256(dest),'crop_method':method})
            if not local: raise ValueError('No decoded frames')
            records.extend(local); methods.update(r['crop_method'] for r in local)
        except (ValueError,RuntimeError,OSError,cv2.error) as error:
            errors.append({'filename':row['filename'],'error':str(error)[:250]})
            # Orphaned partial-video crops stay auditable but never enter training.
        if video_index%20==0 or video_index==len(rows): print(f'Prepared {video_index}/{len(rows)} videos',flush=True); check_budget()
    if not records: raise RuntimeError('No frames prepared; cache is incomplete')
    csv_path=output/'frames.csv'
    with csv_path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    report={'format':'dfdc_cache_v1','complete':not errors,'manifest':str(manifest),'manifest_sha256':sha256(manifest),'official_provenance_verified':preparation.get('official_provenance_verified',False),
            'frames_sha256':sha256(csv_path),'preprocessing':spec,'frames':len(records),'videos':len({r['filename'] for r in records}), 'crop_counts':dict(methods),'errors':errors,
            'split_frame_counts':dict(Counter(r['split'] for r in records)), 'cache_bytes':tree_bytes(output)}
    atomic_json(output/'cache.json',report)
    return report


def load_cache(path,require_official=True):
    path=Path(path).resolve(); report=json.loads((path/'cache.json').read_text(encoding='utf-8'))
    if report.get('format')!='dfdc_cache_v1' or not report.get('complete'): raise ValueError('Cache is incomplete')
    load_manifest(report['manifest'],require_official)
    if sha256(report['manifest'])!=report['manifest_sha256'] or sha256(path/'frames.csv')!=report['frames_sha256']: raise ValueError('Cache/manifest hash mismatch')
    with (path/'frames.csv').open(newline='',encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    if len(rows)!=report['frames']: raise ValueError('Cache frame count mismatch')
    for row in rows:
        file=(path/row['relative_path']).resolve()
        if not file.is_relative_to(path) or not file.is_file() or sha256(file)!=row['sha256']: raise ValueError('Missing/changed/unsafe cached frame')
        if int(row['target'])!=CLASS_TO_IDX[row['label'].lower()]: raise ValueError('Cache class mapping mismatch')
    return rows,report


class JPEGCompression:
    def __call__(self,image):
        import io
        buffer=io.BytesIO(); image.save(buffer,format='JPEG',quality=int(torch.randint(50,96,()).item()))
        buffer.seek(0)
        with Image.open(buffer) as decoded: return decoded.convert('RGB')


class SensorNoise:
    def __call__(self,tensor):
        return (tensor+torch.randn_like(tensor)*(.01+.02*torch.rand(()))).clamp(0,1)


def image_transform(augmentation='none',size=224):
    if augmentation not in {'none','basic','jpeg','noise'}: raise ValueError('Unknown augmentation')
    steps=[transforms.Resize((size,size))]
    if augmentation!='none': steps+=[transforms.RandomHorizontalFlip(.5),transforms.ColorJitter(.1,.1,.05,.02)]
    if augmentation=='jpeg': steps+=[transforms.RandomApply([JPEGCompression()],p=.5)]
    steps+=[transforms.ToTensor()]
    if augmentation=='noise': steps+=[transforms.RandomApply([SensorNoise()],p=.5)]
    steps+=[transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]
    return transforms.Compose(steps)


class FrameDataset(Dataset):
    def __init__(self,cache,rows,split,augmentation='none',size=224):
        self.root=Path(cache); self.rows=[r for r in rows if r['split']==split]
        self.transform=image_transform(augmentation if split=='train' else 'none',size)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(self.root/row['relative_path']) as image: tensor=self.transform(image.convert('RGB'))
        return tensor,int(row['target']),index


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/dfdc/prepared_seed42/manifest.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'data/dfdc/cache_yunet_4')
    parser.add_argument('--faces',choices=['yunet','full_frame','haar'],default='yunet')
    parser.add_argument('--no-align',action='store_true'); parser.add_argument('--frames',type=int,default=4)
    args=parser.parse_args()
    print(json.dumps(prepare_cache(args.manifest,args.output,args.frames,faces=args.faces,align=not args.no_align),indent=2))
