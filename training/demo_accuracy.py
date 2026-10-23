"""Deadline-bound continuation; never overwrites baseline files or activates itself."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from config import PROJECT_ROOT, TRAIN_PATH, VALID_PATH, TEST_PATH
from evaluation.dfdc import binary_metrics
from models.classifier import load_model_checkpoint
from training.data import training_transform, evaluation_transform, _seed_worker
from tools.runtime import configure_runtime


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class IndexedFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index


def utc():
    return datetime.now(timezone.utc).isoformat()


def image_metrics(fake_truth, fake_probabilities):
    """Metric labels use FAKE=positive; model logits retain FAKE=0, REAL=1."""
    metrics = binary_metrics(fake_truth, fake_probabilities)
    tn, fp = metrics['confusion_matrix_real_fake'][0]
    fn, tp = metrics['confusion_matrix_real_fake'][1]
    real_f1 = 2*tn/max(2*tn+fp+fn, 1)
    fake_f1 = 2*tp/max(2*tp+fp+fn, 1)
    metrics.update(real_f1=real_f1, fake_precision=metrics['precision'], fake_f1=fake_f1,
                   weighted_f1=(real_f1*(tn+fp)+fake_f1*(tp+fn))/max(tn+fp+fn+tp, 1),
                   false_positives=fp, false_negatives=fn, true_positives=tp, true_negatives=tn,
                   false_positive_rate=fp/max(tn+fp, 1), false_negative_rate=fn/max(tp+fn, 1),
                   specificity=tn/max(tn+fp, 1), sensitivity=tp/max(tp+fn, 1),
                   classwise={'REAL': {'precision': metrics['real_precision'], 'recall': metrics['real_recall'],
                                       'f1': real_f1, 'support': tn+fp},
                              'FAKE': {'precision': metrics['precision'], 'recall': metrics['recall'],
                                       'f1': fake_f1, 'support': tp+fn}})
    return metrics


def evaluate(model, loader, device, output, split, checkpoint_hash, amp=False, deadline=None):
    model.eval()
    probabilities, truth, indices = [], [], []
    loss_sum, count = 0., 0
    tick = time.perf_counter()
    with torch.inference_mode():
        for images, labels, positions in loader:
            if deadline and time.time() >= deadline:
                raise TimeoutError('Evaluation deadline reached; result is incomplete')
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=amp):
                logits = model(images)
            logits = logits.float()
            loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction='sum'))
            probabilities.extend(logits.softmax(1)[:, 0].cpu().tolist())
            truth.extend((labels == 0).long().cpu().tolist())
            indices.extend(positions.tolist())
            count += len(labels)
    assert count == len(loader.dataset), (count, len(loader.dataset))
    metrics = image_metrics(truth, probabilities)
    metrics.update(loss=loss_sum/count, samples=count, split=split, completed=True,
                   checkpoint_sha256=checkpoint_hash, threshold=.5, positive_class='FAKE',
                   class_to_idx={'fake': 0, 'real': 1}, inference_precision='AMP FP16' if amp else 'FP32',
                   preprocessing='RGB full frame; PIL bilinear 224; ImageNet normalization',
                   seconds=time.perf_counter()-tick, evaluated_utc=utc())
    prediction_file = Path(output).with_suffix('.csv')
    temporary = prediction_file.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['input', 'label', 'fake_probability', 'prediction'])
        writer.writeheader()
        for index, target, probability in zip(indices, truth, probabilities):
            writer.writerow({'input': loader.dataset.samples[index][0], 'label': 'FAKE' if target else 'REAL',
                             'fake_probability': probability, 'prediction': 'FAKE' if probability >= .5 else 'REAL'})
    temporary.replace(prediction_file)
    metrics['predictions_sha256'] = sha(prediction_file)
    write(output, metrics)
    return metrics


def run(args):
    configure_runtime()
    root = args.output.resolve()
    assert root.is_relative_to(PROJECT_ROOT / 'results')
    manifest = json.loads((root/'rollback_manifest.json').read_text(encoding='utf-8'))
    training_deadline = datetime.fromisoformat(manifest['deadlines_utc']['training']).timestamp()
    finish_deadline = datetime.fromisoformat(manifest['deadlines_utc']['finish']).timestamp()
    if args.mode == 'train' and time.time() >= datetime.fromisoformat(manifest['deadlines_utc']['new_experiments']).timestamp():
        raise TimeoutError('New-experiment cutoff reached; baseline remains active')
    write(root/'processes.json', [{'pid': os.getpid(), 'role': args.mode, 'started_utc': utc()}])
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available; baseline remains active')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda')
    source = PROJECT_ROOT/'models/best_resnet18.pth'
    baseline_hash = sha(source)
    expected = next(item['sha256'] for item in manifest['files'] if item['path'] == 'models/best_resnet18.pth')
    assert baseline_hash == expected
    model, saved = load_model_checkpoint(source, device)
    assert saved['class_to_idx'] == {'fake': 0, 'real': 1}
    assert saved['image_size'] == 224 and saved['trainable_backbone_blocks'] == 1
    write(root/'baseline/checkpoint_metadata.json', {key: value for key, value in saved.items() if key != 'model_state_dict'})
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                  worker_init_fn=_seed_worker, persistent_workers=args.workers > 0)
    validation_data = IndexedFolder(VALID_PATH, transform=evaluation_transform())
    assert validation_data.class_to_idx == {'fake': 0, 'real': 1} and len(validation_data) == 20000
    validation = DataLoader(validation_data, shuffle=False, **common)
    if args.mode == 'final-evaluate':
        selected = json.loads((root/'selection.json').read_text(encoding='utf-8'))
        if not selected['improved']:
            print('No selected improvement; test set not opened.', flush=True)
            return
        assert sha(selected['checkpoint']) == selected['checkpoint_sha256']
        # Freeze the selected weights into a separate immutable deployment file.
        # The original checkpoint did not save optimizer state; record the
        # continuation duration separately rather than retaining its old total.
        history = json.loads((root/'history.json').read_text(encoding='utf-8'))
        candidate_saved = torch.load(selected['checkpoint'], map_location='cpu', weights_only=True)
        continued_seconds = sum(row['seconds'] for row in history if row['original_epoch'] <= selected['epoch'])
        original_seconds = float(saved.get('training_seconds_so_far', 0.))
        candidate_saved.update(baseline_training_seconds=original_seconds,
                               continuation_training_seconds=continued_seconds,
                               training_seconds_so_far=original_seconds+continued_seconds,
                               continuation_additional_epochs=selected['epoch']-int(saved['epoch']),
                               preprocessing_protocol={'rgb':True,'crop':'full_frame','resize':'PIL BILINEAR',
                                                       'size':224,'normalization':'ImageNet','threshold':.5})
        frozen = root/'selected_final.pth'
        if frozen.exists() or (root/'selection_frozen.json').exists():
            raise FileExistsError('Final selection is already frozen; refusing another final evaluation')
        torch.save(candidate_saved, frozen)
        reloaded = torch.load(frozen, map_location='cpu', weights_only=True)
        assert all(torch.equal(value, reloaded['model_state_dict'][key]) for key,value in candidate_saved['model_state_dict'].items())
        selected.update(checkpoint=str(frozen),checkpoint_sha256=sha(frozen),frozen_utc=utc(),
                        continuation_seconds=continued_seconds,original_training_seconds=original_seconds)
        write(root/'selection.json',selected)
        write(root/'selection_frozen.json',selected)
        model, selected_saved = load_model_checkpoint(selected['checkpoint'], device)
        selected_hash = sha(selected['checkpoint'])
        assert selected_hash == selected['checkpoint_sha256']
        metrics = evaluate(model, validation, device, root/'candidate_validation_fp32.json', 'validation', selected_hash,
                           deadline=finish_deadline-180)
        baseline = json.loads((root/'baseline_validation_fp32.json').read_text(encoding='utf-8'))
        if metrics['balanced_accuracy'] <= baseline['balanced_accuracy']:
            write(root/'final_status.json', {'improved': False, 'reason': 'FP32 validation did not improve', 'test_evaluated': False})
            return
        # Selection is now fixed. This is the sole candidate test invocation.
        claim = root/'candidate_test.claim'
        with claim.open('x', encoding='utf-8') as stream:
            stream.write(selected_hash + '\n')
        test_data = IndexedFolder(TEST_PATH, transform=evaluation_transform())
        assert len(test_data) == 20000 and test_data.class_to_idx == validation_data.class_to_idx
        test = DataLoader(test_data, shuffle=False, **common)
        test_metrics = evaluate(model, test, device, root/'candidate_test_fp32.json', 'test', selected_hash,
                                deadline=finish_deadline-90)
        write(root/'final_status.json', {'improved': True, 'completed': True, 'test_evaluated': True,
                                       'checkpoint': selected['checkpoint'], 'checkpoint_sha256': selected_hash,
                                       'validation': metrics, 'test': test_metrics})
        print(json.dumps({'validation': metrics, 'test': test_metrics}), flush=True)
        return
    train_data = IndexedFolder(TRAIN_PATH, transform=training_transform())
    assert len(train_data) == 100000 and train_data.class_to_idx == validation_data.class_to_idx
    train = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(42), **common)
    registration = {'started_utc': utc(), 'baseline_checkpoint': str(source), 'baseline_sha256': baseline_hash,
                    'approach': 'Continue original ResNet-18 weights; fresh optimizer because original has no optimizer state',
                    'train_images': len(train_data), 'validation_images': len(validation_data), 'test_policy': 'final selected candidate only',
                    'trainable_backbone_blocks': 1, 'augmentation': 'unchanged original flip/color/rotation transform',
                    'learning_rate': 3e-5, 'weight_decay': 1e-4, 'scheduler': 'ReduceLROnPlateau factor=.5 patience=1',
                    'selection': 'full 20,000-image validation balanced accuracy; validation loss tie-breaker; threshold .5',
                    'max_additional_epochs': args.epochs, 'batch_size': args.batch_size, 'workers': args.workers,
                    'gpu': torch.cuda.get_device_name(), 'gpu_free_bytes': torch.cuda.mem_get_info()[0],
                    'alternatives': 'Cached EfficientNet-B0 and DeiT-Tiny need new training; no cached stronger ResNet/Inception. Original epochs measured 297-362 seconds. Prefer continuation within deadline.'}
    write(root/'registration.json', registration)
    baseline = evaluate(model, validation, device, root/'baseline_validation_fp32.json', 'validation', baseline_hash,
                        deadline=training_deadline-300)
    print('BASELINE ' + json.dumps(baseline), flush=True)
    model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1)
    scaler = torch.amp.GradScaler('cuda')
    best = (baseline['balanced_accuracy'], -baseline['loss'])
    history = []
    last_epoch_seconds = 362.
    for epoch in range(1, args.epochs+1):
        remaining = training_deadline-time.time()
        if remaining < last_epoch_seconds*1.12+75:
            print(f'Not starting epoch {epoch}: {remaining:.1f}s left before training stop.', flush=True)
            break
        tick = time.perf_counter()
        model.train(); loss_sum=0.; correct=0; count=0
        for batch, (images, labels, positions) in enumerate(train):
            if time.time() >= training_deadline-45:
                write(root/'deadline_stop.json', {'epoch': epoch, 'partial_training_images': count, 'utc': utc()})
                print('Training deadline: partial epoch discarded; best completed checkpoint retained.', flush=True)
                break
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                logits = model(images); loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach())*len(labels)
            correct += int((logits.argmax(1)==labels).sum()); count += len(labels)
            if batch % 50 == 0:
                progress={'phase':'training','epoch':epoch,'images':count,'total':len(train_data),
                          'images_per_second':count/max(time.perf_counter()-tick,1e-6),'utc':utc(),
                          'remaining_training_seconds':training_deadline-time.time()}
                write(root/'progress.json',progress)
                print(json.dumps(progress),flush=True)
        if count != len(train_data):
            break
        # Full validation after every completed epoch; no favorable subset selection.
        metrics = evaluate(model, validation, device, root/f'epoch_{epoch}_validation.json', 'validation',
                           'in_memory_epoch_'+str(epoch), amp=False, deadline=training_deadline)
        scheduler.step(metrics['loss'])
        row={'additional_epoch':epoch,'original_epoch':int(saved['epoch'])+epoch,'train_loss':loss_sum/count,
             'train_accuracy':correct/count,'validation':metrics,'learning_rate':optimizer.param_groups[0]['lr'],
             'seconds':time.perf_counter()-tick}
        history.append(row); write(root/'history.json',history)
        score=(metrics['balanced_accuracy'],-metrics['loss'])
        if score>best:
            best=score
            candidate={**{key:value for key,value in saved.items() if key!='model_state_dict'},
                       'model_state_dict':{key:value.detach().cpu() for key,value in model.state_dict().items()},
                       'epoch':int(saved['epoch'])+epoch,'valid_loss':metrics['loss'],'valid_accuracy':metrics['accuracy'],
                       'continued_from_sha256':baseline_hash,'continuation_config':registration}
            target=root/'candidate_best.pth'; temporary=root/'candidate_best.pth.tmp'
            torch.save(candidate,temporary); temporary.replace(target)
            write(root/'selection.json',{'improved':metrics['balanced_accuracy']>baseline['balanced_accuracy'],
                    'checkpoint':str(target),'checkpoint_sha256':sha(target),'epoch':int(saved['epoch'])+epoch,
                    'validation':metrics,'baseline':baseline,'selected_on':'validation only','test_evaluated':False})
        last_epoch_seconds=row['seconds']
        print('EPOCH '+json.dumps(row),flush=True)
    if not (root/'selection.json').exists():
        write(root/'selection.json',{'improved':False,'baseline':baseline,'reason':'No completed epoch improved full validation accuracy'})
    assert sha(source)==baseline_hash
    write(root/'training_finished.json',{'utc':utc(),'completed_epochs':len(history),'baseline_unchanged':True})
    print('TRAINING FINISHED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['train','final-evaluate'],default='train')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'failure.json',{'type':type(error).__name__,'error':str(error),'utc':utc(),'baseline_active':True})
        raise
