"""Resumable continuation of the two existing image identities on predefined splits.

Run with the existing dfenv: python -B -m training.image_continuation --phase train
No historical checkpoint, experiment summary, or video artifact is overwritten.
"""
import argparse
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from tools.runtime import configure_runtime, memory_snapshot
configure_runtime()

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from config import PROJECT_ROOT, TRAIN_PATH, VALID_PATH, TEST_PATH
from inference.contracts import checkpoint_digest
from models.classifier import load_model_checkpoint, parameter_counts
from models.transfer import CLASS_TO_IDX, load_transfer_checkpoint
from training.data import inspect_dataset, training_transform, evaluation_transform
from training.demo_accuracy import IndexedFolder, evaluate, sha, utc, write

ROOT = PROJECT_ROOT / 'results/current_image_models'
IDENTITIES = ('R18_FT_20261021', 'E02_efficientnet_b0')
PROTOCOL = {
    'max_epoch': 10, 'patience': 3, 'min_delta': 1e-4,
    'selection': 'validation balanced accuracy; ties within 1e-12 use lower FP32 validation loss',
    'early_stopping': 'validation loss must decrease by more than min_delta; independent of model selection',
    'optimizer': 'AdamW', 'backbone_lr': 3e-5, 'classifier_lr': 1e-4,
    'resnet_classifier_lr': 3e-5, 'weight_decay': 1e-4,
    'scheduler': 'ReduceLROnPlateau', 'scheduler_factor': .5, 'scheduler_patience': 1,
    'loss': 'CrossEntropyLoss', 'amp_training': True, 'validation_precision': 'FP32',
    'preprocessing': 'RGB full frame; PIL bilinear 224; ImageNet normalization',
    'augmentation': 'unchanged training.data.training_transform: flip, mild color jitter, rotation 4 degrees',
    'threshold': .5, 'class_to_idx': CLASS_TO_IDX,
    'test_policy': 'freeze both selections and promotion eligibility on validation before any final test evaluation; repeated historical test, not a newly sealed benchmark',
    'augmentation_rng': 'deterministic per sample and epoch; main CPU/CUDA RNG and shuffle generator restored',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_checkpoint(path, payload):
    """Validate a weights-only round trip before an atomic same-volume replace."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checked = torch.load(temporary, map_location='cpu', weights_only=True)
    if checked.get('class_to_idx') != CLASS_TO_IDX:
        raise ValueError('Checkpoint class mapping changed')
    for key, value in payload['model_state_dict'].items():
        if not torch.equal(value.cpu(), checked['model_state_dict'][key]):
            raise ValueError('Checkpoint round-trip mismatch: ' + key)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite checkpoint tensor: ' + key)
    digest = sha(temporary)
    pending = path.with_suffix(path.suffix + '.pending.json')
    write(pending, {'sha256': digest})
    temporary.replace(path)
    write(path.with_suffix(path.suffix + '.sha256.json'), {'sha256': digest})
    pending.unlink(missing_ok=True)
    return digest


def checked_training_state(path):
    """Verify state bytes, including recovery between atomic file/hash writes."""
    digest = sha(path)
    sidecar = path.with_suffix(path.suffix + '.sha256.json')
    pending = path.with_suffix(path.suffix + '.pending.json')
    valid_hashes = [read(p)['sha256'] for p in [sidecar, pending] if p.exists()]
    if digest not in valid_hashes:
        raise ValueError('Resumable training-state SHA-256 mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if pending.exists():
        write(sidecar, {'sha256': digest})
        pending.unlink()
    return state


def load_architecture(path, identity, device='cpu'):
    model, saved = (load_model_checkpoint(path, torch.device('cpu')) if identity == IDENTITIES[0]
                    else load_transfer_checkpoint(path, 'cpu'))
    if saved.get('class_to_idx') != CLASS_TO_IDX or saved.get('image_size') != 224:
        raise ValueError('Initialization violates image/class contract')
    if identity == IDENTITIES[0]:
        if saved.get('model_name') != 'resnet18' or saved.get('trainable_backbone_blocks') != 1:
            raise ValueError('ResNet continuation must retain layer4 and the existing head')
    else:
        if saved.get('model_spec', {}).get('backbone') != 'efficientnet_b0':
            raise ValueError('E02 must be the image EfficientNet-B0, not a video encoder')
        model.set_trainable(2)
    return model.to(device).eval(), saved


def historical_hashes():
    paths = subprocess.check_output([
        'git', 'ls-files', 'results/dfdc_unified_20261019', 'results/champion.json',
        'results/execution_20261019', 'results/demo_accuracy_20261021_040741',
        'docs/figures/demo_accuracy_20261021'], cwd=PROJECT_ROOT, text=True).splitlines()
    return {path: sha(PROJECT_ROOT/path) for path in paths if (PROJECT_ROOT/path).is_file()}


def preflight_log(root, report):
    summary = {'started_utc': report['started_utc'], 'executable': report['executable'],
               'python': report['python'], 'torch': report['torch'], 'cuda': report['cuda'],
               'gpu': report['gpu'], 'vram_bytes': report['vram_bytes'],
               'disk_free_bytes': report['disk_free_bytes'], 'memory': report['memory'],
               'split_counts': {key: {field: value[field] for field in ['total', 'fake', 'real']}
                                for key, value in report['dataset']['splits'].items()},
               'sources': {key: {field: value[field] for field in ['checkpoint', 'sha256', 'epoch', 'resume_mode']}
                           for key, value in report['sources'].items()}}
    path = root/'preflight_summary.log'
    if not path.exists():
        path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print('PREFLIGHT ' + json.dumps(summary), flush=True)


def audit(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'preflight.json').exists():
        report = read(root/'preflight.json')
        for item in report['sources'].values():
            if sha(item['checkpoint']) != item['sha256']:
                raise ValueError('Original initialization changed after preflight')
        preflight_log(root, report)
        return report
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU training is forbidden')
    from inference.demo_candidate import active_image_candidate
    active = active_image_candidate()
    if not active or active['id'] != IDENTITIES[0]:
        raise ValueError('The verified active October 21 ResNet record is required')
    e02 = read(PROJECT_ROOT/'results/execution_20261019/E02_efficientnet_b0/summary.json')
    sources = {
        IDENTITIES[0]: {'checkpoint': active['checkpoint'], 'sha256': active['checkpoint_sha256']},
        IDENTITIES[1]: {'checkpoint': e02['checkpoint'], 'sha256': e02['checkpoint_sha256']},
    }
    for identity, item in sources.items():
        path = Path(item['checkpoint'])
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError('Missing/corrupt local initialization: ' + identity)
        model, saved = load_architecture(path, identity)
        if int(saved['epoch']) != (6 if identity == IDENTITIES[0] else 3):
            raise ValueError('Unexpected initial epoch: ' + identity)
        item.update(epoch=int(saved['epoch']), bytes=path.stat().st_size,
                    optimizer_state_available='optimizer_state_dict' in saved,
                    resume_mode='weight-only continuation; original checkpoints have no optimizer/scheduler/scaler state',
                    initialization='existing ImageNet-pretrained, image-trained checkpoint',
                    previous_training_images=100000 if identity == IDENTITIES[0] else 12000,
                    metadata={k: v for k, v in saved.items() if k != 'model_state_dict'})
        del model, saved
    dataset = inspect_dataset(save_report=False)
    for split, expected in [('train', 100000), ('valid', 20000), ('test', 20000)]:
        actual = dataset['splits'][split]
        if actual['total'] != expected or actual['fake'] != expected//2 or actual['real'] != expected//2:
            raise ValueError('Unexpected predefined split counts: ' + split)
    manifests = {}
    for split, folder in [('train', TRAIN_PATH), ('validation', VALID_PATH), ('test', TEST_PATH)]:
        data = IndexedFolder(folder)
        if data.class_to_idx != CLASS_TO_IDX:
            raise ValueError('Unexpected mapping in ' + split)
        path = root/f'{split}_manifest.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['relative_path', 'target', 'bytes', 'mtime_ns'])
            for name, target in data.samples:
                stat = Path(name).stat()
                writer.writerow([Path(name).relative_to(folder).as_posix(), target, stat.st_size, stat.st_mtime_ns])
        manifests[split] = {'path': str(path), 'sha256': sha(path), 'samples': len(data)}
    # These backups are local rollback assets, not new experiment identities.
    backups = []
    paths = [Path(item['checkpoint']) for item in sources.values()] + [
        PROJECT_ROOT/'configs/runtime/active_image_model.json',
        PROJECT_ROOT/'results/demo_accuracy_20261021_040741/deployment.json',
        PROJECT_ROOT/'results/execution_20261019/E02_efficientnet_b0/summary.json',
        PROJECT_ROOT/'results/champion.json', PROJECT_ROOT/'README.md',
        PROJECT_ROOT/'inference/catalog.py', PROJECT_ROOT/'inference/demo_candidate.py', PROJECT_ROOT/'app.py']
    for path in paths:
        destination = root/'rollback'/path.relative_to(PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha(path) != sha(destination):
            raise ValueError('Rollback copy failed integrity: ' + str(path))
        backups.append({'path': str(path), 'backup': str(destination), 'sha256': sha(path)})
    report = {'started_utc': utc(), 'python': sys.version, 'executable': sys.executable,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
              'free_gpu_bytes': torch.cuda.mem_get_info()[0], 'memory': memory_snapshot(),
              'disk_free_bytes': shutil.disk_usage(PROJECT_ROOT).free,
              'dataset': dataset, 'manifests': manifests, 'sources': sources,
              'backups': backups, 'historical_hashes': historical_hashes(), 'protocol': PROTOCOL}
    write(root/'preflight.json', report)
    preflight_log(root, report)
    print('PREFLIGHT ' + json.dumps({k: report[k] for k in ['gpu', 'torch', 'disk_free_bytes', 'memory']}), flush=True)
    print('SPLITS ' + json.dumps(dataset['splits']), flush=True)
    print('SOURCES ' + json.dumps({k: {x: v[x] for x in ['checkpoint', 'sha256', 'epoch', 'resume_mode']} for k, v in sources.items()}), flush=True)
    return report


class EpochFolder(IndexedFolder):
    """Reproducible augmentations even across persistent-worker resume boundaries."""
    def __init__(self, path):
        super().__init__(path, transform=training_transform())
        self.epoch = mp.Value('i', 0)

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            # Seed only the CPU transform generator. torch.manual_seed() also
            # reseeds CUDA when num_workers=0, perturbing dropout and adding work.
            torch.random.default_generator.manual_seed(42 + self.epoch.value * len(self) + index)
            return super().__getitem__(index)


class SplitFolder(Dataset):
    """Share two persistent workers across train and validation to bound RAM."""
    def __init__(self, decode_threads=4):
        self.training = EpochFolder(TRAIN_PATH)
        self.validation = IndexedFolder(VALID_PATH, transform=evaluation_transform())
        self.mode = mp.Value('i', 1)  # Validation baseline first.
        self.epoch = self.training.epoch
        self.threaded = False
        self.pool = None
        self.decode_threads = decode_threads

    @property
    def current(self):
        return self.training if self.mode.value == 0 else self.validation

    @property
    def samples(self):
        return self.current.samples

    def __len__(self):
        return len(self.current)

    def __getitem__(self, index):
        return self.current[index]

    def _resize(self, index):
        from PIL import Image
        path, target = self.current.samples[index]
        return self.current.loader(path).resize((224, 224), Image.Resampling.BILINEAR), target, index

    def _augmentation_parameters(self, index):
        from torchvision import transforms
        parts = self.training.transform.transforms
        expected = [transforms.Resize, transforms.RandomHorizontalFlip, transforms.RandomApply,
                    transforms.RandomRotation, transforms.ToTensor, transforms.Normalize]
        if len(parts) != len(expected) or any(not isinstance(t, kind) for t, kind in zip(parts, expected)):
            raise ValueError('Threaded augmentation requires the unchanged verified training transform')
        jitter = parts[2].transforms[0]
        if len(parts[2].transforms) != 1 or not isinstance(jitter, transforms.ColorJitter):
            raise ValueError('Unexpected training color augmentation')
        # Sample the same torchvision RNG calls in their original order, then
        # run deterministic PIL operations in parallel without shared RNG races.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42+self.epoch.value*len(self.training)+index)
            flip = bool(torch.rand(1) < parts[1].p)
            color = None if bool(parts[2].p < torch.rand(1)) else jitter.get_params(
                jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)
            angle = parts[3].get_params(parts[3].degrees)
        return flip, color, angle

    def _augmented_item(self, item):
        from torchvision.transforms import functional as functional
        index, (flip, color, angle) = item
        image, target, _ = self._resize(index)
        if flip:
            image = functional.hflip(image)
        if color is not None:
            order, brightness, contrast, saturation, hue = color
            operations = [functional.adjust_brightness, functional.adjust_contrast,
                          functional.adjust_saturation, functional.adjust_hue]
            factors = [brightness, contrast, saturation, hue]
            for position in order.tolist():
                if factors[position] is not None:
                    image = operations[position](image, factors[position])
        rotation = self.training.transform.transforms[3]
        image = functional.rotate(image, angle, rotation.interpolation, rotation.expand, rotation.center, rotation.fill)
        tensor, normalize = self.training.transform.transforms[-2:]
        return normalize(tensor(image)), target, index

    def __getitems__(self, indices):
        if not self.threaded:
            return [self[index] for index in indices]
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=self.decode_threads, thread_name_prefix='image-decode')
        if self.mode.value == 1:
            return list(self.pool.map(self.validation.__getitem__, indices))
        parameters = [self._augmentation_parameters(index) for index in indices]
        return list(self.pool.map(self._augmented_item, zip(indices, parameters)))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


class SplitSampler(Sampler):
    def __init__(self, dataset, generator):
        self.dataset = dataset
        self.generator = generator

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.dataset.mode.value == 0:
            return iter(torch.randperm(len(self.dataset), generator=self.generator).tolist())
        return iter(range(len(self.dataset)))


class PrefetchLoader:
    """Prepare one pinned CPU batch ahead without duplicating PyTorch processes."""
    def __init__(self, base):
        self.base = base
        self.dataset = base.dataset

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        iterator = iter(self.base)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='image-prefetch') as pool:
            future = pool.submit(next, iterator, None)
            while True:
                batch = future.result()
                if batch is None:
                    return
                future = pool.submit(next, iterator, None)
                yield batch


class ThreadedEvaluationFolder(IndexedFolder):
    def __init__(self, path, decode_threads=4):
        super().__init__(path, transform=evaluation_transform())
        self.pool = ThreadPoolExecutor(max_workers=decode_threads, thread_name_prefix='evaluation-decode')

    def __getitems__(self, indices):
        return list(self.pool.map(super().__getitem__, indices))

    def close(self):
        self.pool.shutdown(wait=True)


def loader(dataset, batch_size, workers, training=False, generator=None):
    options = {'prefetch_factor': 2} if workers else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=True, generator=generator, **options)


def better(metrics, best):
    difference = metrics['balanced_accuracy'] - best['balanced_accuracy']
    return difference > 1e-12 or (abs(difference) <= 1e-12 and metrics['loss'] < best['loss'])


def loss_progress(loss, best, bad, min_delta=1e-4):
    return (loss, 0) if loss < best - min_delta else (best, bad + 1)


def epoch_numbers(identity, completed):
    if identity not in IDENTITIES or completed < (6 if identity == IDENTITIES[0] else 0) or completed > 10:
        raise ValueError('Invalid continuation epoch counter')
    return range(completed + 1, 11)


def optimizer_for(model, identity):
    if identity == IDENTITIES[0]:
        groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 3e-5}]
    else:
        groups = [{'params': list(model.head.parameters()), 'lr': 1e-4},
                  {'params': [p for p in model.features.parameters() if p.requires_grad], 'lr': 3e-5}]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def smoke(identity, source, batch_sizes=(32, 64)):
    """Real CUDA backward tests; compare throughput before choosing 64."""
    outcomes = []
    for batch in batch_sizes:
        model, _ = load_architecture(source, identity, 'cuda')
        model.to(memory_format=torch.channels_last).train()
        optimizer = optimizer_for(model, identity)
        scaler = torch.amp.GradScaler('cuda')
        inputs = torch.randn(batch, 3, 224, 224, device='cuda').to(memory_format=torch.channels_last)
        targets = torch.arange(batch, device='cuda') % 2
        torch.cuda.reset_peak_memory_stats()
        try:
            times = []
            for step in range(7):
                torch.cuda.synchronize(); tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    loss = nn.functional.cross_entropy(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite smoke loss')
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                torch.cuda.synchronize()
                if step >= 2:
                    times.append(time.perf_counter() - tick)
            outcomes.append({'batch_size': batch, 'passed': True, 'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                             'images_per_second': batch / float(np.mean(times))})
        except torch.cuda.OutOfMemoryError:
            outcomes.append({'batch_size': batch, 'passed': False, 'reason': 'CUDA OOM'})
        finally:
            if 'loss' in locals():
                del loss
            del model, optimizer, scaler, inputs, targets
            gc.collect(); torch.cuda.empty_cache()
    passed = [r for r in outcomes if r['passed'] and r['peak_vram_bytes'] < .65 * torch.cuda.get_device_properties(0).total_memory]
    if not passed:
        if batch_sizes[-1] > 8:
            return smoke(identity, source, (16, 8))
        raise RuntimeError('No safe CUDA microbatch, refusing CPU fallback')
    chosen = passed[0]
    for row in passed[1:]:
        if row['images_per_second'] > 1.05 * chosen['images_per_second']:
            chosen = row
    return {'identity': identity, 'outcomes': outcomes, 'chosen_batch_size': chosen['batch_size'],
            'effective_batch_size': max(32, chosen['batch_size']), 'synthetic_throughput_scope': 'GPU-only smoke; actual epoch throughput recorded separately'}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def historical_history(identity):
    if identity != IDENTITIES[0]:
        return []
    with (PROJECT_ROOT/'outputs/training_history.csv').open(newline='', encoding='utf-8') as stream:
        rows = [{'epoch': int(r['epoch']), 'train_loss': float(r['train_loss']), 'train_accuracy': float(r['train_accuracy']),
                 'validation': {'accuracy': float(r['valid_accuracy']), 'loss': float(r['valid_loss'])},
                 'seconds': float(r['epoch_seconds']), 'stage': 'original 100k training', 'historical': True}
                for r in csv.DictReader(stream)]
    for row in read(PROJECT_ROOT/'results/demo_accuracy_20261021_040741/history.json'):
        rows.append({'epoch': row['original_epoch'], 'train_loss': row['train_loss'], 'train_accuracy': row['train_accuracy'],
                     'validation': row['validation'], 'seconds': row['seconds'], 'stage': 'October 21 continuation', 'historical': True})
    if [r['epoch'] for r in rows] != list(range(1, 7)):
        raise ValueError('Historical ResNet epochs are missing or duplicated')
    return rows


def train_one(identity, preflight, root, workers, threaded=False, decode_threads=4):
    out = root/identity; out.mkdir(exist_ok=True)
    source = preflight['sources'][identity]
    if (out/'training_finished.json').exists():
        return read(out/'training_finished.json')
    source_path = source['checkpoint']
    smoke_path = out/'cuda_smoke.json'
    if not smoke_path.exists():
        write(smoke_path, smoke(identity, source_path))
    smoke_result = read(smoke_path)
    batch_size = smoke_result['chosen_batch_size']
    accumulation = max(1, smoke_result['effective_batch_size']//batch_size)
    model, metadata = load_architecture(source_path, identity, 'cuda')
    model.to(memory_format=torch.channels_last)
    train_data = SplitFolder(decode_threads)
    train_data.threaded = threaded
    if threaded:
        workers = 0
    valid_data = train_data.validation
    generator = torch.Generator().manual_seed(42)
    worker_generator = torch.Generator().manual_seed(1042)
    options = {'prefetch_factor': 1} if workers else {}
    training = DataLoader(train_data, batch_size=batch_size, sampler=SplitSampler(train_data, generator),
                         num_workers=workers, persistent_workers=workers > 0, pin_memory=True,
                         generator=worker_generator, **options)
    if threaded:
        training = PrefetchLoader(training)
    validation = training  # One shared persistent worker pool, never four workers.
    optimizer = optimizer_for(model, identity)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1)
    scaler = torch.amp.GradScaler('cuda')
    base_metadata = {key: value for key, value in metadata.items() if key != 'model_state_dict'}
    del metadata
    base_metadata.update(logical_identity=identity, class_to_idx=CLASS_TO_IDX, fine_tune_blocks=2 if identity == IDENTITIES[1] else 1,
                         dataset_splits=preflight['manifests'], training_images=100000, hyperparameters=PROTOCOL,
                         warm_start={k: v for k, v in source.items() if k != 'metadata'})
    history = historical_history(identity)
    start_epoch = 6 if identity == IDENTITIES[0] else 0
    selected_path = Path(source_path)
    selected_epoch = start_epoch
    bad = 0; training_seconds = 0.; best_loss = float('inf')
    last_path = out/'last_training_state.pt'
    if last_path.exists():
        state = checked_training_state(last_path)
        if state['dataset_splits'] != preflight['manifests'] or state['hyperparameters'] != PROTOCOL:
            raise ValueError('Resume protocol/dataset changed')
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        start_epoch = state['completed_epoch']; bad = state['early_stopping_counter']
        best_loss = state['best_early_stopping_loss']; best = state['selected_validation']
        history = state['history']; training_seconds = state['stage_training_seconds']
        selected_path = Path(state['selected_checkpoint']); selected_epoch = state['selected_epoch']
        if sha(selected_path) != state['selected_checkpoint_sha256']:
            raise ValueError('Resume best-checkpoint hash mismatch')
        torch.set_rng_state(state['torch_rng_state'])
        torch.cuda.set_rng_state_all(state['cuda_rng_states'])
        random.setstate(state['python_rng_state'])
        generator.set_state(state['loader_generator_state'])
        events_path = out/'resume_events.json'
        events = read(events_path) if events_path.exists() else []
        events.append({'utc': utc(), 'committed_epoch': start_epoch, 'next_epoch': start_epoch+1,
                       'state_sha256': sha(last_path), 'optimizer_restored': True,
                       'scheduler_restored': True, 'scaler_restored': True,
                       'cpu_cuda_rng_restored': True, 'selected_epoch': selected_epoch})
        write(events_path, events)
        print(f'RESUME {identity}: committed epoch {start_epoch}, bad={bad}', flush=True)
    else:
        best = verified_evaluation(model, validation, out/'baseline_validation_fp32.json', 'validation', source['sha256'])
        best_loss = best['loss']
    # A full-validation baseline of E02 is retained as a conservative candidate,
    # but its historical 12k epochs never enter the new 100k stage numbering.
    torch.cuda.reset_peak_memory_stats()
    for epoch in epoch_numbers(identity, start_epoch):
        if bad >= PROTOCOL['patience']:
            break
        tick = time.perf_counter()
        train_data.mode.value = 0
        train_data.epoch.value = epoch
        generator.manual_seed(42 + epoch)
        used_lrs = [group['lr'] for group in optimizer.param_groups]
        model.train(); count = 0; loss_sum = 0.; correct = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets, _) in enumerate(training):
            images = images.to('cuda', non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to('cuda', non_blocking=True)
            with torch.autocast('cuda'):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            group_first = (step//accumulation)*accumulation
            group_samples = min(accumulation*batch_size, len(train_data)-group_first*batch_size)
            scaler.scale(loss * len(targets)/group_samples).backward()
            if (step+1) % accumulation == 0 or step+1 == len(training):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)
            if step % 100 == 0:
                progress = {'identity': identity, 'phase': 'training', 'epoch': epoch, 'images': count, 'total': len(train_data),
                            'images_per_second': count/max(time.perf_counter()-tick, 1e-6), 'utc': utc()}
                write(root/'progress.json', progress); print(json.dumps(progress), flush=True)
        if count != 100000:
            raise ValueError('Incomplete training epoch')
        train_data.mode.value = 1
        metrics = evaluate(model, validation, torch.device('cuda'), out/f'epoch_{epoch}_validation_fp32.json', 'validation', f'in_memory_{identity}_epoch_{epoch}')
        scheduler.step(metrics['loss'])
        best_loss, bad = loss_progress(metrics['loss'], best_loss, bad, PROTOCOL['min_delta'])
        row = {'epoch': epoch, 'stage': '100k continuation' if identity == IDENTITIES[0] else '100k full-dataset stage',
               'train_loss': loss_sum/count, 'train_accuracy': correct/count, 'training_images': count,
               'validation': metrics, 'learning_rates_used': used_lrs,
               'learning_rates_next': [group['lr'] for group in optimizer.param_groups],
               'early_stopping_counter': bad, 'seconds': time.perf_counter()-tick,
               'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'batch_size': batch_size,
               'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(),
               'gradient_accumulation': accumulation, 'workers': workers,
               'loader_backend': f'{decode_threads} image transform threads in main process' if threaded else 'persistent DataLoader processes',
               'historical': False}
        training_seconds += row['seconds']; history.append(row)
        weights = cpu_state(model)
        if better(metrics, best):
            selected_path = out/f'selected_epoch_{epoch}.pth'
            selected_epoch = epoch; best = metrics
            payload = {**base_metadata, 'model_state_dict': weights, 'epoch': epoch,
                       'full_dataset_stage_epoch': epoch if identity == IDENTITIES[1] else None,
                       'valid_loss': metrics['loss'], 'valid_accuracy': metrics['accuracy']}
            atomic_checkpoint(selected_path, payload)
        state = {**base_metadata, 'model_state_dict': weights, 'epoch': epoch, 'completed_epoch': epoch,
                 'full_dataset_stage_epoch': epoch if identity == IDENTITIES[1] else None,
                 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                 'scaler_state_dict': scaler.state_dict(), 'early_stopping_counter': bad,
                 'best_early_stopping_loss': best_loss, 'selected_validation': best,
                 'selected_checkpoint': str(selected_path), 'selected_checkpoint_sha256': sha(selected_path),
                 'selected_epoch': selected_epoch, 'history': history, 'stage_training_seconds': training_seconds,
                 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
                 'python_rng_state': random.getstate(), 'loader_generator_state': generator.get_state()}
        atomic_checkpoint(last_path, state)
        write(out/'history.json', history)
        write(out/'selection.json', {'identity': identity, 'checkpoint': str(selected_path), 'checkpoint_sha256': sha(selected_path),
                                    'epoch': selected_epoch, 'validation': best, 'selected_on': PROTOCOL['selection'], 'test_evaluated': False})
        print('EPOCH ' + json.dumps({'identity': identity, **row}), flush=True)
        del state, weights
    model.load_state_dict(torch.load(selected_path, map_location='cpu', weights_only=True)['model_state_dict'])
    model.eval()
    stage_rows = [r for r in history if not r.get('historical', False)]
    total, trainable = parameter_counts(model)
    final = {'identity': identity, 'completed': True, 'start_epoch': 7 if identity == IDENTITIES[0] else 1,
             'end_epoch': stage_rows[-1]['epoch'] if stage_rows else start_epoch, 'selected_epoch': selected_epoch,
             'selected_checkpoint': str(selected_path), 'selected_checkpoint_sha256': sha(selected_path),
             'selected_validation': best, 'early_stopping_triggered': bad >= PROTOCOL['patience'],
             'training_seconds': training_seconds, 'peak_vram_bytes': max([r['peak_vram_bytes'] for r in stage_rows], default=0),
             'parameters': total, 'trainable_parameters': trainable, 'checkpoint_bytes': selected_path.stat().st_size,
             'physical_batch_size': batch_size, 'effective_batch_size': batch_size*accumulation,
             'loader_configuration': {'worker_processes': workers, 'decode_threads': decode_threads if threaded else 0,
                                      'prefetched_cpu_batches': 1 if threaded else workers,
                                      'backend': 'threaded PIL decode plus CPU prefetch' if threaded else 'shared persistent DataLoader workers'},
             'warm_start': {k: v for k, v in source.items() if k != 'metadata'}, 'protocol': PROTOCOL,
             'history': history, 'finished_utc': utc()}
    final['resume_events'] = read(out/'resume_events.json') if (out/'resume_events.json').exists() else []
    write(out/'selection.json', {'identity': identity, 'checkpoint': str(selected_path), 'checkpoint_sha256': sha(selected_path),
                                'epoch': selected_epoch, 'validation': best, 'selected_on': PROTOCOL['selection'], 'test_evaluated': False})
    write(out/'training_finished.json', final)
    train_data.close()
    del model, optimizer, scheduler, scaler, training, validation, train_data, valid_data
    gc.collect(); torch.cuda.empty_cache()
    return final


def verified_evaluation(model, data_loader, path, split, digest):
    """Reuse only complete, matching prediction evidence after interruptions."""
    if path.exists() and path.with_suffix('.csv').exists():
        existing = read(path)
        if (existing.get('completed') and existing.get('checkpoint_sha256') == digest
                and existing.get('samples') == len(data_loader.dataset)
                and existing.get('predictions_sha256') == sha(path.with_suffix('.csv'))):
            return existing
    return evaluate(model, data_loader, torch.device('cuda'), path, split, digest)


def profile_loading(root):
    from torch.utils.data import Subset
    dataset = EpochFolder(VALID_PATH)
    records = []
    for workers in [0, 1, 2]:
        data_loader = loader(Subset(dataset, list(range(768))), 64, workers)
        tick = time.perf_counter(); count = 0
        for images, labels, _ in data_loader:
            if count == 0:
                warmup = time.perf_counter()-tick
                tick = time.perf_counter()
            else:
                pass
            count += len(labels)
        seconds = time.perf_counter()-tick
        record = {'workers': workers, 'startup_seconds': warmup, 'steady_images_per_second': (count-64)/seconds,
                  'memory': memory_snapshot(), 'measured_images': count}
        records.append(record)
        print('LOADER PROFILE ' + json.dumps(record), flush=True)
        del data_loader; gc.collect()
    best = max(records, key=lambda r: r['steady_images_per_second'])
    write(root/'loader_profile.json', {'records': records, 'chosen_workers': best['workers'],
                                     'scope': 'CPU loading/augmentation only; validation images used only for throughput probe, no optimizer steps'})


def profile_threaded_loading(root):
    from torch.utils.data import Subset
    dataset = SplitFolder(); dataset.mode.value = 0; dataset.epoch.value = 7
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:2048].tolist()
    # Verify exact preprocessing/augmentation equivalence on actual train files.
    reference = [dataset[i] for i in indices[:8]]
    dataset.threaded = True
    optimized = dataset.__getitems__(indices[:8])
    if any(a[1:] != b[1:] or not torch.equal(a[0], b[0]) for a, b in zip(reference, optimized)):
        raise ValueError('Threaded PIL preprocessing changed training tensors')
    data_loader = loader(Subset(dataset, indices), 64, 0)
    tick = time.perf_counter(); count = 0
    for images, labels, _ in data_loader:
        count += len(labels)
    report = {'backend': 'four decode threads; zero worker subprocesses', 'samples': count,
              'images_per_second': count/(time.perf_counter()-tick), 'memory': memory_snapshot(),
              'exact_training_tensor_equivalence': True}
    dataset.close()
    write(root/'threaded_loader_profile.json', report)
    print('THREADED LOADER ' + json.dumps(report), flush=True)


def profile_thread_scaling(root):
    from torch.utils.data import Subset
    dataset = SplitFolder(); dataset.mode.value = 0; dataset.epoch.value = 3
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:2048].tolist()
    reference = [dataset[i] for i in indices[:16]]
    dataset.threaded = True
    # Warm the same files once before comparing thread counts, avoiding a
    # systematic cold-disk advantage for a later measurement.
    for _ in loader(Subset(dataset, indices), 64, 0):
        pass
    records = []
    for threads in [4, 6, 8, 4]:
        dataset.close(); dataset.decode_threads = threads
        actual = dataset.__getitems__(indices[:16])
        if any(a[1:] != b[1:] or not torch.equal(a[0], b[0]) for a, b in zip(reference, actual)):
            raise ValueError('Thread scaling changed training tensors')
        tick = time.perf_counter(); count = 0
        for _, targets, _ in loader(Subset(dataset, indices), 64, 0):
            count += len(targets)
        record = {'threads': threads, 'images_per_second': count/(time.perf_counter()-tick),
                  'samples': count, 'memory': memory_snapshot(), 'exact_tensor_equivalence': True}
        records.append(record)
        print('THREAD SCALING ' + json.dumps(record), flush=True)
    baseline = max(r['images_per_second'] for r in records if r['threads'] == 4)
    chosen = max(records, key=lambda r: r['images_per_second'])
    threads = chosen['threads'] if chosen['images_per_second'] > baseline*1.10 else 4
    write(root/'thread_scaling_profile.json', {'records': records, 'chosen_threads': threads,
                                             'protocol': 'same 2,048 train files; prewarmed; fixed epoch/sample augmentation; no optimizer/model steps'})
    dataset.close()


def benchmark(model, validation_data):
    """Identical resident-input FP32 forward protocol for both architectures."""
    model.eval().to(memory_format=torch.contiguous_format)
    torch.cuda.reset_peak_memory_stats()
    protocol = {'precision': 'FP32; TF32 disabled', 'batch_size': 32, 'warmups': 10, 'repeats': 30,
                'synchronization': 'torch.cuda.synchronize before and after each forward',
                'scope': 'resident-input model forward; excludes decoding, preprocessing, transfer, XAI',
                'input': 'first 32 predefined validation images, unchanged evaluation preprocessing'}
    inputs = torch.stack([validation_data[i][0] for i in range(32)]).to('cuda')
    results = {}
    for batch in [32, 1]:
        tensor = inputs[:batch].contiguous()
        elapsed = []
        with torch.inference_mode():
            for step in range(protocol['warmups'] + protocol['repeats']):
                torch.cuda.synchronize(); tick = time.perf_counter()
                model(tensor)
                torch.cuda.synchronize()
                if step >= protocol['warmups']:
                    elapsed.append(time.perf_counter()-tick)
        seconds = float(np.mean(elapsed))
        results[str(batch)] = {'mean_batch_ms': seconds*1000, 'median_batch_ms': float(np.median(elapsed))*1000,
                               'ms_per_image': seconds*1000/batch, 'images_per_second': batch/seconds,
                               'timed_batch_seconds': elapsed}
    return {'protocol': protocol, 'batches': results, 'peak_vram_bytes': torch.cuda.max_memory_allocated()}


def final_evaluate(root, preflight, workers, threaded=False, decode_threads=4):
    if not (root/'training_complete.json').exists():
        raise ValueError('Both sequential training stages must finish before final selection')
    frozen_path = root/'selection_frozen.json'
    if not frozen_path.exists():
        frozen = {'frozen_utc': utc(), 'selected_on': PROTOCOL['selection'], 'test_used_for_selection': False, 'models': {}}
        # Both decisions and their eligibility are frozen before test is loaded.
        for identity in IDENTITIES:
            folder = root/identity
            finished = read(folder/'training_finished.json')
            source = Path(finished['selected_checkpoint'])
            if sha(source) != finished['selected_checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed before freeze')
            baseline = read(folder/'baseline_validation_fp32.json')
            selected = finished['selected_validation']
            if better(baseline, selected):
                raise ValueError('Selected model regressed against its full-validation baseline')
            target = folder/'selected_final.pth'
            payload = torch.load(source, map_location='cpu', weights_only=True)
            origin = preflight['sources'][identity]
            selected_rows = [row for row in finished['history']
                             if not row.get('historical', False) and row['epoch'] <= finished['selected_epoch']]
            selected_seconds = sum(row['seconds'] for row in selected_rows)
            previous_seconds = origin['metadata'].get('training_seconds_so_far')
            if previous_seconds is None and identity == IDENTITIES[1]:
                previous_seconds = read(PROJECT_ROOT/'results/execution_20261019/E02_efficientnet_b0/summary.json')['training_seconds']
            # Legacy continuation fields describe the October 21 stage. Preserve
            # them under the warm-start metadata and bind current fields to the
            # immediate source, rather than accidentally claiming an epoch-2 resume.
            payload.update(warm_start_checkpoint_metadata=origin['metadata'],
                           continued_from_sha256=origin['sha256'], continuation_config=PROTOCOL,
                           continuation_training_seconds=selected_seconds,
                           baseline_training_seconds=previous_seconds,
                           training_seconds_so_far=(previous_seconds+selected_seconds) if previous_seconds is not None else None,
                           continuation_additional_epochs=len(selected_rows),
                           training_images=100000 if finished['selected_epoch'] > 0 else origin['previous_training_images'],
                           epoch=finished['selected_epoch'])
            if identity == IDENTITIES[1]:
                payload['full_dataset_stage_epoch'] = finished['selected_epoch']
            payload.update(selection_frozen_utc=frozen['frozen_utc'], selected_on=PROTOCOL['selection'],
                           stage_summary={k: v for k, v in finished.items() if k != 'history'},
                           training_history=finished['history'])
            digest = atomic_checkpoint(target, payload)
            frozen['models'][identity] = {'checkpoint': str(target), 'checkpoint_sha256': digest,
                                         'selected_epoch': finished['selected_epoch'], 'validation': selected,
                                         'promotion_eligible': not better(baseline, selected),
                                         'improved_on_validation': better(selected, baseline)}
        write(frozen_path, frozen)
    frozen = read(frozen_path)
    validation_data = ThreadedEvaluationFolder(VALID_PATH, decode_threads) if threaded else IndexedFolder(VALID_PATH, transform=evaluation_transform())
    validation_loader = loader(validation_data, 64, 0 if threaded else workers)
    if threaded:
        validation_loader = PrefetchLoader(validation_loader)
    # Reconfirm both frozen checkpoints in FP32 before opening the test split.
    for identity in IDENTITIES:
        item = frozen['models'][identity]; folder = root/identity
        if sha(item['checkpoint']) != item['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint hash mismatch')
        model, _ = load_architecture(item['checkpoint'], identity, 'cuda')
        model.to(memory_format=torch.channels_last)
        write(root/'progress.json', {'phase': 'final_validation', 'identity': identity, 'total': 20000, 'utc': utc()})
        print('FINAL FP32 VALIDATION '+identity, flush=True)
        validation = verified_evaluation(model, validation_loader, folder/'candidate_validation_fp32.json', 'validation', item['checkpoint_sha256'])
        if abs(validation['balanced_accuracy']-item['validation']['balanced_accuracy']) > 1e-12 or abs(validation['loss']-item['validation']['loss']) > 1e-6:
            raise ValueError('Frozen FP32 validation does not reproduce selection')
        write(folder/'inference_benchmark.json', benchmark(model, validation_data))
        print('VALIDATION RESULT '+json.dumps({'identity': identity, 'accuracy': validation['accuracy'],
                                              'balanced_accuracy': validation['balanced_accuracy'], 'loss': validation['loss']}), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    write(root/'promotion_decisions_frozen.json', {'utc': utc(), 'models': frozen['models'], 'test_used': False})
    test_data = ThreadedEvaluationFolder(TEST_PATH, decode_threads) if threaded else IndexedFolder(TEST_PATH, transform=evaluation_transform())
    test_loader = loader(test_data, 64, 0 if threaded else workers)
    if threaded:
        test_loader = PrefetchLoader(test_loader)
    from training.demo_accuracy import image_metrics
    reference_path = PROJECT_ROOT/'results/execution_20261019/E00_validation/validation_predictions.csv'
    with reference_path.open(newline='', encoding='utf-8') as stream:
        references = list(csv.DictReader(stream))
    for identity in IDENTITIES:
        item = frozen['models'][identity]; folder = root/identity
        model, _ = load_architecture(item['checkpoint'], identity, 'cuda')
        model.to(memory_format=torch.channels_last)
        write(root/'progress.json', {'phase': 'final_test', 'identity': identity, 'total': 20000, 'utc': utc()})
        print('FINAL FP32 TEST '+identity+'; selections already frozen', flush=True)
        test = verified_evaluation(model, test_loader, folder/'candidate_test_fp32.json', 'test', item['checkpoint_sha256'])
        print('TEST RESULT '+json.dumps({'identity': identity, 'accuracy': test['accuracy'],
                                        'balanced_accuracy': test['balanced_accuracy'], 'roc_auc': test['roc_auc']}), flush=True)
        validation = read(folder/'candidate_validation_fp32.json')
        with (folder/'candidate_validation_fp32.csv').open(newline='', encoding='utf-8') as stream:
            by_path = {str(Path(r['input']).resolve()).casefold(): r for r in csv.DictReader(stream)}
        common = [by_path[str(Path(r.get('path', r.get('input'))).resolve()).casefold()] for r in references]
        if len(common) != 2000 or any(a['label'] != b['label'] for a, b in zip(common, references)):
            raise ValueError('Common historical validation cohort changed')
        metrics = image_metrics([int(r['label']=='FAKE') for r in common], [float(r['fake_probability']) for r in common])
        metrics.update(samples=2000, split='validation', completed=True, checkpoint_sha256=item['checkpoint_sha256'],
                       inference_precision='FP32', scope='historical common 2,000-image validation subset; reporting only, not epoch selection')
        write(folder/'common_validation_fp32.json', metrics)
        write(folder/'final_status.json', {'completed': True, 'test_evaluated': True, **item,
                                         'validation': validation, 'test': test, 'common_validation': metrics,
                                         'checkpoint_bytes': Path(item['checkpoint']).stat().st_size,
                                         'test_scope': 'repeat evaluation of historically used predefined test split'})
        del model; gc.collect(); torch.cuda.empty_cache()
    write(root/'evaluation_complete.json', {'completed': True, 'utc': utc(), 'selection_frozen': str(frozen_path)})
    write(root/'progress.json', {'phase': 'final_evaluation_complete', 'utc': utc()})
    if threaded:
        validation_data.close(); test_data.close()


def run(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    preflight = audit(args.output)
    if args.phase == 'audit':
        return
    if args.phase == 'profile':
        profile_loading(args.output)
        return
    if args.phase == 'profile-threaded':
        profile_threaded_loading(args.output)
        return
    if args.phase == 'profile-threads':
        profile_thread_scaling(args.output)
        return
    if args.phase == 'evaluate':
        final_evaluate(args.output, preflight, args.workers, args.threaded_loader, args.decode_threads)
        return
    if args.phase == 'smoke':
        for identity in IDENTITIES:
            folder = args.output/identity; folder.mkdir(exist_ok=True)
            write(folder/'cuda_smoke.json', smoke(identity, preflight['sources'][identity]['checkpoint']))
        return
    for identity in IDENTITIES:
        while True:
            try:
                train_one(identity, preflight, args.output, args.workers, args.threaded_loader, args.decode_threads)
                break
            except torch.cuda.OutOfMemoryError:
                # A partially completed epoch is retried from the last atomic
                # state, never counted as complete. Keep the effective batch.
                gc.collect(); torch.cuda.empty_cache()
                smoke_path = args.output/identity/'cuda_smoke.json'
                sizing = read(smoke_path)
                old = sizing['chosen_batch_size']
                if old <= 8:
                    raise
                sizing['chosen_batch_size'] = old//2
                sizing.setdefault('oom_recovery', []).append({'previous_batch': old, 'new_batch': old//2, 'utc': utc()})
                write(smoke_path, sizing)
                print(f'CUDA OOM: resuming {identity} with microbatch {old//2} and accumulation', flush=True)
    write(args.output/'training_complete.json', {'completed': True, 'identities': list(IDENTITIES), 'utc': utc()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['audit', 'smoke', 'profile', 'profile-threaded', 'profile-threads', 'train', 'evaluate'], default='train')
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threaded-loader', action='store_true', help='Use in-process image transform threads to reduce Windows RAM pressure')
    parser.add_argument('--decode-threads', type=int, choices=[4, 6, 8], default=4)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as error:
        arguments.output.mkdir(parents=True, exist_ok=True)
        write(arguments.output/'failure.json', {'type': type(error).__name__, 'error': str(error), 'utc': utc(),
                                               'deployment_unchanged': True, 'resume': 'rerun the same command; last committed epoch is retained'})
        raise
