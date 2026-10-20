"""Limited domain-aware frozen-head continuation with a matched DFDC-only control.

Only existing train/validation data are used. Neither supplied sanity image nor
any test pixels/features enter this experiment. Its registration is saved before
feature extraction or optimization; historical frozen runs are immutable.
"""
import copy
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from evaluation.dfdc import binary_metrics
from models.unified import FeatureHead, read_checkpoint
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import atomic_json, check_budget
from tools.runtime import configure_runtime
from training.dfdc_cache import make_cropper
from training.unified_experiment import feature_data, load_encoder, pixels, calibrate
from training.video_protocol import RUN, SPEC

OUTPUT = RUN.parent/'debug_refinement_20261019/domain_pilot'
_local = threading.local()


def prepare(row):
    if sha256(row['path']) != row['sha256']:
        raise ValueError('Image content changed')
    if not hasattr(_local, 'crop'):
        _local.crop = make_cropper(SPEC)
    with Image.open(row['path']) as image:
        rgb = np.asarray(image.convert('RGB'))
    crop, status = _local.crop(rgb[:, :, ::-1].copy())
    array = np.asarray(Image.fromarray(crop[:, :, ::-1].copy()).resize((224,224), Image.Resampling.BILINEAR))
    return array, status


def extract_images(rows, output, device):
    cache = output/'image_features.pt'
    if cache.exists():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        if saved['sha256'] != [row['sha256'] for row in rows]:
            raise ValueError('Pilot cache manifest mismatch')
        return saved['values']
    encoder = load_encoder('efficientnet_b0', device)
    features, statuses = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool, torch.inference_mode():
        for start in range(0, len(rows), 32):
            prepared = list(pool.map(prepare, rows[start:start+32]))
            tensor = pixels(np.stack([pair[0] for pair in prepared])).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                features.append(encoder(tensor).float().cpu())
            statuses.extend(pair[1] for pair in prepared)
            if start % 512 == 0:
                print(f'Domain pilot feature extraction {start}/{len(rows)}', flush=True)
    values = torch.cat(features)
    torch.save({'values': values, 'sha256': [row['sha256'] for row in rows], 'statuses': statuses}, cache)
    atomic_json(output/'extraction.json', {'images': len(rows), 'seconds': time.perf_counter()-started,
        'preprocessing': SPEC, 'crop_status_counts': {status: statuses.count(status) for status in set(statuses)}})
    del encoder
    torch.cuda.empty_cache()
    return values


def evaluate(head, video_features, video_y, image_features, image_y):
    with torch.inference_mode():
        video_logits = head(video_features).cpu()
        image_logits = head(image_features[:, None]).cpu()
    return {'video': binary_metrics(video_y.cpu().numpy(), video_logits.softmax(-1)[:,1].numpy()),
            'image': binary_metrics(image_y.cpu().numpy(), image_logits.softmax(-1)[:,1].numpy())}, video_logits, image_logits


def run(output=OUTPUT, per_class=1000, epochs=16, domain_weight=.5):
    configure_runtime(); torch.set_num_threads(2); torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('The registered pilot requires the existing CUDA device')
    if not 0<domain_weight<1:
        raise ValueError('Both domains must have positive weights')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    check_budget(200*1024**2)
    manifest = RUN.parent/'execution_20261019/dataset/manifest.csv'
    with manifest.open(newline='', encoding='utf-8') as stream:
        all_rows = list(csv.DictReader(stream))
    rng = np.random.default_rng(42)
    train = []
    for label in ['REAL', 'FAKE']:
        eligible = sorted([row for row in all_rows if row['split']=='train' and row['label']==label], key=lambda row:row['path'])
        indices = sorted(rng.choice(len(eligible), per_class, replace=False))
        train += [eligible[index] for index in indices]
    validation = [row for row in all_rows if row['split']=='validation']
    rows = train+validation
    assert all(row['split'] in {'train','validation'} for row in rows)
    assert not {row['sha256'] for row in train} & {row['sha256'] for row in validation}
    config = json.loads((RUN/'S2/config.json').read_text(encoding='utf-8'))
    head, saved = read_checkpoint(RUN/'S2/best.pt')
    atomic_json(output/'registration.json', {'image_train_per_class':per_class, 'image_validation':len(validation),
        'epochs':epochs, 'domain_weight_image':domain_weight, 'domain_weight_video':1-domain_weight,
        'batch_per_domain':32, 'steps_per_epoch':26, 'learning_rate':.0005, 'weight_decay':.01,
        'seed':42, 'encoder':'existing frozen ImageNet EfficientNet-B0; no encoder updates',
        'initial_head_sha256':sha256(RUN/'S2/best.pt'), 'image_manifest_sha256':sha256(manifest),
        'video_manifest_sha256':sha256(RUN/'manifest.csv'), 'normalization':'original S2 train-only feature statistics for both arms',
        'selection':'harmonic mean of separate image and video validation balanced accuracies; macro-F1 tie break',
        'deployment_rule':'Must match both existing modality champions within 1 percentage point and improve one by at least 1 point. Otherwise research only.',
        'test_access':False, 'sanity_image_access':False, 'class_mapping':{'REAL':0,'FAKE':1},
        'control':'Same initial S2 head, video samples, optimizer, steps and validation selection; no image loss',
        'training_policy':'Independent inverse-class-frequency replacement sampling within each domain; average domain losses, not concatenated frames'})
    atomic_json(output/'image_manifest.json', rows)
    images = extract_images(rows, output, device)
    mean, std = saved['feature_mean'], saved['feature_std']
    train_x = ((images[:len(train)]-mean)/std).to(device)
    valid_x = ((images[len(train):]-mean)/std).to(device)
    train_y = torch.tensor([int(row['label']=='FAKE') for row in train], device=device)
    valid_y = torch.tensor([int(row['label']=='FAKE') for row in validation], device=device)
    video_train, video_y, _ = feature_data(config, 'train')
    video_valid, video_valid_y, _ = feature_data(config, 'validation')
    video_train = ((video_train-mean)/std).to(device); video_valid = ((video_valid-mean)/std).to(device)
    video_y = video_y.to(device); video_valid_y = video_valid_y.to(device)
    head = head.to(device)
    baseline, _, _ = evaluate(head, video_valid, video_valid_y, valid_x, valid_y)
    atomic_json(output/'S2_baseline_validation.json', baseline)
    results = {}
    video_weights = torch.tensor([1/float((video_y==int(y)).sum()) for y in video_y.cpu()], dtype=torch.float64)
    image_weights = torch.tensor([1/float((train_y==int(y)).sum()) for y in train_y.cpu()], dtype=torch.float64)
    for name, image_weight in [('dfdc_only_control',0.), ('mixed_domains',domain_weight)]:
        directory = output/name; directory.mkdir()
        model = copy.deepcopy(head).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
        generator_video = torch.Generator().manual_seed(42)
        generator_image = torch.Generator().manual_seed(43)
        best = (-1., -1.); history = []; best_state = None; best_metrics = None
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train(); losses=[]
            for step in range(26):
                indices = torch.multinomial(video_weights, 32, replacement=True, generator=generator_video).to(device)
                image_indices = torch.multinomial(image_weights, 32, replacement=True, generator=generator_image).to(device)
                optimizer.zero_grad(set_to_none=True)
                video_loss = nn.functional.cross_entropy(model(video_train[indices]),video_y[indices])
                if image_weight:
                    image_loss = nn.functional.cross_entropy(model(train_x[image_indices,None]),train_y[image_indices])
                    loss = (1-image_weight)*video_loss+image_weight*image_loss
                else:
                    loss = video_loss
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            model.eval()
            metrics, video_logits, image_logits = evaluate(model,video_valid,video_valid_y,valid_x,valid_y)
            va, ia = metrics['video']['balanced_accuracy'],metrics['image']['balanced_accuracy']
            score = (2*va*ia/max(va+ia,1e-12), (metrics['video']['macro_f1']+metrics['image']['macro_f1'])/2)
            history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':metrics})
            if score>best:
                best=score;best_state=copy.deepcopy(model.state_dict());best_metrics=metrics
                torch.save({'format':'domain_pilot_v1','head_state':best_state,'feature_mean':mean,'feature_std':std,
                            'class_to_idx':{'real':0,'fake':1},'epoch':epoch+1,'registration':str(output/'registration.json')},directory/'best.pt')
                torch.save({'video':video_logits,'image':image_logits},directory/'validation_logits.pt')
            atomic_json(directory/'history.json',history)
            print(f"{name} epoch {epoch+1}: video BA {va:.4f}; image BA {ia:.4f}",flush=True)
        results[name]={'validation':best_metrics,'selection_harmonic_balanced_accuracy':best[0],
                       'training_seconds':time.perf_counter()-started,'checkpoint':str(directory/'best.pt'),
                       'checkpoint_sha256':sha256(directory/'best.pt')}
        best_logits=torch.load(directory/'validation_logits.pt',weights_only=True)
        results[name]['calibration']={'image':calibrate(valid_y.cpu().numpy(),best_logits['image']),
                                     'video':calibrate(video_valid_y.cpu().numpy(),best_logits['video'])}
        for unit in ['image','video']:
            results[name]['calibration'][unit].update(evaluation_unit=unit,checkpoint_sha256=results[name]['checkpoint_sha256'],method=f'scalar temperature on {unit} aggregate logits/log probabilities')
        # Both probability files retain original input identity and separate units.
        atomic_json(directory/'summary.json',results[name])
    results['baseline_S2']=baseline
    image_champion=.969; video_champion=.9481707317073171
    metrics=results['mixed_domains']['validation']
    ia,va=metrics['image']['balanced_accuracy'],metrics['video']['balanced_accuracy']
    accepted=(ia>=image_champion-.01 and va>=video_champion-.01 and (ia>=image_champion+.01 or va>=video_champion+.01))
    results['deployed']=accepted
    results['deployment_reason']='Meets registered rule' if accepted else 'Does not meet registered joint-domain champion rule; retain existing image and video defaults'
    atomic_json(output/'comparison.json',results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--per-class',type=int,default=1000)
    parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--domain-weight',type=float,default=.5)
    args=parser.parse_args()
    run(args.output,args.per_class,args.epochs,args.domain_weight)
