"""Small controlled attribution checks; these do not improve classifier accuracy."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from explainability.gradcam import GradCAM
from inference.media import MediaPredictor
from tools.dfdc_storage import ROOT, atomic_json
from training.dfdc_cache import load_cache


def evaluate_map(model,layer,tensor,class_index,method='gradcam',fraction=.15,seed=42):
    if tensor.shape[0]!=1 or not 0<fraction<1: raise ValueError('One image and a mask fraction in (0,1) required')
    generator=GradCAM(model,layer)
    try:
        cam=generator.generate(tensor,class_index,method)
        flipped=generator.generate(tensor.flip(-1),class_index,method)[:,::-1]
    finally: generator.close()
    k=max(1,int(cam.size*fraction));rng=np.random.default_rng(seed)
    salient=np.argsort(cam.ravel(),kind='stable')[-k:]
    def score(indices=None):
        value=tensor.clone()
        if indices is not None:
            mask=torch.zeros(cam.size,dtype=torch.bool,device=tensor.device);mask[indices.copy()]=True
            value=value.masked_fill(mask.reshape(1,1,*cam.shape),0.)
        with torch.inference_mode(): return float(model(value).float().softmax(1)[0,class_index])
    clean=score();deletion=clean-score(salient)
    random_drops=[clean-score(rng.choice(cam.size,k,replace=False)) for _ in range(5)]
    correlation=float(np.corrcoef(cam.ravel(),flipped.ravel())[0,1]) if cam.std()>1e-8 and flipped.std()>1e-8 else None
    return {'method':method,'target_class':'FAKE' if class_index==0 else 'REAL','mask_fraction':fraction,'clean_probability':clean,
            'salient_deletion_drop':deletion,'random_deletion_drop_mean':float(np.mean(random_drops)),
            'deletion_advantage_over_random':deletion-float(np.mean(random_drops)),'flip_map_correlation':correlation,
            'seed':seed,'random_masks':5,'limitation':'Mean-filled masks can be out of distribution. Negative drops and undefined correlations are retained. Flip stability is not proof of faithfulness.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'data/dfdc/cache_yunet_4');parser.add_argument('--limit',type=int,default=6)
    args=parser.parse_args()
    rows,_=load_cache(args.cache);predictor=MediaPredictor(args.checkpoint)
    selected=[r for r in rows if r['split']=='validation'][:min(args.limit,12)];results=[]
    for row in selected:
        with Image.open(args.cache/row['relative_path']) as image: tensor=predictor.transform(image.convert('RGB')).unsqueeze(0).to(predictor.device)
        with torch.inference_mode(): index=int(predictor.model(tensor).argmax(1))
        for method in ['gradcam','gradcam++']:
            results.append({'filename':row['filename'],'frame_index':row['frame_index'],**evaluate_map(predictor.model,predictor.layer,tensor,index,method)})
    atomic_json(args.checkpoint.parent/'xai_validation.json',{'evaluation_split':'validation','results':results,'classification_metrics_unchanged':True})
    print(f'Saved {len(results)} validation attribution checks')
