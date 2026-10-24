"""Read-only checkpoint experiments for the 2026-10-24 XAI audit.

Run with python -u -m tools.audit_xai_end_to_end. Media and tensors remain in
the ignored results directory; this does not train or alter deployment pointers.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

from tools.runtime import configure_runtime
configure_runtime()
import numpy as np
import torch
from PIL import Image, ImageDraw
from config import PROJECT_ROOT, DATASET_PATH
from inference.catalog import catalog, create_predictor, methods_for
from inference.contracts import checkpoint_digest
from inference.media import save_media_result
from explainability.methods import integrated_gradients, occlusion_sensitivity, lime_explanation, check_attribution

OUT = PROJECT_ROOT / 'results/xai_audit_20261024'
SAVED = Path.home() / 'OneDrive/Pictures/Saved Pictures'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return checkpoint_digest(path)


def model_state(p):
    modules = [v for v in vars(p).values() if isinstance(v, torch.nn.Module)] + getattr(p, 'encoders', [])
    h = hashlib.sha256()
    for module in modules:
        h.update(str(module.training).encode())
        for key, value in module.state_dict().items():
            h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run():
    if (OUT/'experiments.json').exists():
        raise FileExistsError('Use a new --output directory to preserve audit evidence')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    inventory = {kind: catalog(kind, kind == 'Image') for kind in ['Image', 'Video']}
    environment = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
        'platform': platform.platform(), 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ['torch','torchvision','streamlit','timm','numpy','Pillow','opencv-python','scikit-learn','lime']},
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'catalog': inventory}
    dump(OUT/'environment.json', environment)
    video_rows = list(csv.DictReader((PROJECT_ROOT/'results/dfdc_unified_20261019/manifest.csv').open()))
    videos = [next(r for r in video_rows if r['split']=='test' and r['label']==label) for label in ['REAL','FAKE']]
    images = [DATASET_PATH/'test'/label/name for label, name in [('real','00001.jpg'),('fake','00276TOPP4.jpg')]]
    inputs = [{'path':str(p), 'label':None, 'source':'optional local samples; name is not verified ground truth'} for p in sorted(SAVED.iterdir()) if p.suffix.lower() in {'.jpg','.jpeg','.png','.mp4','.mov'}] if SAVED.is_dir() else []
    inputs += [{'path':str(p), 'label':p.parent.name.upper(), 'source':'supplied Real vs Fake Faces test folder; historical test reused, identity disjointness unverified'} for p in images]
    inputs += [{'path':r['path'], 'label':r['label'], 'source':'local DFDC metadata, source-grouped test', 'manifest_sha256':r['sha256']} for r in videos]
    for item in inputs:
        item['sha256']=digest(item['path'])
        if item.get('manifest_sha256'): assert item['manifest_sha256']==item['sha256']
        matches=[r for r in video_rows if r['sha256']==item['sha256']]
        if matches:
            item['label']=matches[0]['label']; item['matched_video_manifest']=matches[0]
    dump(OUT/'inputs.json', inputs)
    cases=[]
    image_ids=['R18_FT_20261021','E02_efficientnet_b0']
    for item in inputs:
        video=Path(item['path']).suffix.lower() in {'.mp4','.mov'}
        ids=['S6'] if video else [image_ids[0]]
        if item['path'] in [str(p) for p in images]: ids=image_ids
        for identifier in ids: cases.append((item,identifier,'Video' if video else 'Image'))
    # Transformer and combined architecture coverage, same metadata-labeled clip.
    for identifier in ['S4','S5','S8']:
        cases.append((next(i for i in inputs if i['path']==videos[1]['path']),identifier,'Video'))
    results=[]
    for number,(item,identifier,kind) in enumerate(cases):
        key=f'{number:02d}_{Path(item["path"]).stem}_{identifier}'
        print('START',key,flush=True)
        record=next(r for r in inventory[kind] if r['id']==identifier)
        p=create_predictor(record)
        before=model_state(p)
        result=p.predict_video(item['path'],method=None) if kind=='Video' else p.predict_image(Image.open(item['path']),method=None)
        prepared=result['prepared_frames'] if kind=='Video' else result['prepared_input'][None]
        position=int(np.argmax([r['fake_probability'] if result['prediction']=='FAKE' else 1-r['fake_probability'] for r in result['sampled_frames']])) if kind=='Video' else 0
        x=p.tensor(prepared[position:position+1]) if hasattr(p,'tensor') else p.transform(Image.fromarray(prepared[0])).unsqueeze(0).to(p.device)
        context=torch.from_numpy(result['feature_context']).to(p.device) if kind=='Video' else None
        view=p.attribution_model(context,position,kind=='Image') if hasattr(p,'attribution_model') else p.model
        target=int(result['prediction']=='FAKE') if hasattr(p,'attribution_model') else int(result['prediction']=='REAL')
        with torch.no_grad(): logits=view(x).float(); clean=float(logits.softmax(-1)[0,target])
        evidence={'key':key,'input':item,'model':identifier,'kind':kind,'prediction':result['prediction'],
            'fake_probability':result['fake_probability'],'real_probability':result['real_probability'],
            'correct':None if item['label'] is None else result['prediction']==item['label'],
            'position':position,'explanation_target_index':target,'explanation_clean_target_probability':clean,
            'prediction_vs_explanation_probability_difference':clean-result['confidence'],'methods':{},
            'crop_method':result.get('crop_method'), 'frame_rows':result.get('sampled_frames'), 'contributions':result.get('contributions')}
        explanations={}
        for method in methods_for(record):
            tick=time.perf_counter()
            try:
                value=p.explain_result(result,method,result['prediction'],**({'frame_indices':[position]} if kind=='Video' else {}))
                e=value[0] if isinstance(value,list) else value
                assert e['cam'].shape==(224,224) and np.isfinite(e['cam']).all()
                diagnostic=check_attribution(view,x,target,e['cam']) if method in {'gradcam','integrated_gradients','occlusion','lime'} else None
                evidence['methods'][method]={'details':e['details'],'deletion':diagnostic,'seconds':time.perf_counter()-tick,'coverage':e['coverage']}
                explanations[method]=e
                print('METHOD',key,method,round(time.perf_counter()-tick,2),flush=True)
            except Exception as error:
                evidence['methods'][method]={'error':repr(error)}
                print('ERROR',key,method,repr(error),flush=True)
        # Controlled spatial masks in actual input coordinates; no semantic claim.
        perturbations={}
        for name,(y1,y2,x1,x2) in {'central':(56,168,56,168),'top_left':(0,64,0,64),'border':(0,32,0,224),'lower_central':(128,192,64,160)}.items():
            changed=x.clone(); changed[:,:,y1:y2,x1:x2]=0
            with torch.no_grad(): altered=view(changed).float()
            perturbations[name]={'bounds_yxyx':[y1,y2,x1,x2],'target_probability_drop':clean-float(altered.softmax(-1)[0,target]),
                'target_margin_drop':float((logits[0,target]-logits[0,1-target])-(altered[0,target]-altered[0,1-target]))}
        evidence['coordinate_mean_masks']=perturbations
        # Stability and raw signed-vs-magnitude IG diagnostics on two images.
        if identifier==image_ids[0] and Path(item['path']).name in {'fake.jpg','real.jpeg'}:
            stability={}
            for steps,baseline in [(32,'mean'),(64,'mean'),(128,'mean'),(64,'blur')]:
                cam,details=integrated_gradients(view,x,target,steps,max_steps=steps,baseline=baseline)
                stability[f'{steps}_{baseline}']=details
            base=torch.zeros_like(x); nodes,weights=np.polynomial.legendre.leggauss(64); total=torch.zeros_like(x)
            for j in range(0,64,8):
                point=(x*((x.new_tensor(nodes[j:j+8])+1)/2).reshape(-1,1,1,1)).detach().requires_grad_(True)
                gradient=torch.autograd.grad(view(point)[:,target].sum(),point)[0]
                total+=(gradient*x.new_tensor(weights[j:j+8]/2).reshape(-1,1,1,1)).sum(0,keepdim=True)
            raw=(x*total).detach().cpu().numpy()[0]
            np.savez_compressed(OUT/f'{key}_ig_raw.npz',raw=raw,signed=raw.sum(0),absolute_magnitude=np.abs(raw).sum(0))
            stability['channel_cancellation_abs_signed_over_abs_channels']=float(np.abs(raw.sum(0)).sum()/max(np.abs(raw).sum(),1e-20))
            evidence['ig_stability']=stability
            evidence['occlusion_stability']={}
            for patch in [16,32,56]:
                _,details=occlusion_sensitivity(view,x,target,patch)
                evidence['occlusion_stability'][str(patch)]=details
            evidence['lime_stability']={}
            maps=[]
            for seed in [42,43,44]:
                cam,details=lime_explanation(view,x,target,seed=seed)
                maps.append(cam.ravel()); evidence['lime_stability'][str(seed)]=details
            evidence['lime_map_correlations']=np.corrcoef(maps).tolist()
        with torch.no_grad(): after=view(x).float()
        evidence['score_unchanged']=bool(torch.equal(logits,after))
        evidence['model_state_unchanged']=before==model_state(p)
        result['audit_explanations']=explanations
        save_media_result(result,OUT/key)
        # Compact contact sheet of all actual maps for visual inspection.
        sheet=Image.new('RGB',(224*3,250*len(explanations)), 'white'); draw=ImageDraw.Draw(sheet)
        for row,(method,e) in enumerate(explanations.items()):
            draw.text((4,row*250+4),method,fill='black')
            for col,field in enumerate(['original','heatmap','overlay']): sheet.paste(Image.fromarray(e[field]),(col*224,row*250+24))
        sheet.save(OUT/f'{key}_gallery.jpg')
        results.append(evidence); dump(OUT/'experiments.json',results)
        del p,view,result,explanations
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print('DONE',key,flush=True)
    dump(OUT/'checkpoint_integrity_after.json', {r['checkpoint']:digest(r['checkpoint'])==r['checkpoint_sha256'] for rows in inventory.values() for r in rows})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--samples',type=Path,default=SAVED,help='Optional additional images/videos; names are not ground-truth labels')
    args=parser.parse_args()
    OUT=args.output.resolve()
    SAVED=args.samples.resolve()
    run()
