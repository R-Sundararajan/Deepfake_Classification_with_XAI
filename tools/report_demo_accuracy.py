"""Write the measured handoff; update README only after verified activation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from config import PROJECT_ROOT


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def table_row(model, split, metrics):
    if not metrics:
        return f'| {model} | {split} | not evaluated | — | — | — | — | — |'
    return '| '+ ' | '.join([model,split,f"{metrics['samples']:,}",f"{metrics['accuracy']:.3%}",
        f"{metrics.get('balanced_accuracy',metrics['accuracy']):.3%}",f"{metrics['precision']:.3%}",
        f"{metrics['recall']:.3%}",f"{metrics.get('f1',metrics.get('f1_score')):.5f}"])+' |'


def run(root,activated):
    root=root.resolve()
    manifest=load(root/'rollback_manifest.json')
    baseline=load(root/'baseline_validation_fp32.json')
    historical=load(PROJECT_ROOT/'results/execution_20261019/E00_test/metrics.json')
    status=load(root/'final_status.json') or {}
    selected=load(root/'selection.json') or {}
    deployment=load(root/'deployment.json')
    history=load(root/'history.json') or []
    if activated:
        assert status.get('completed') and deployment and deployment['verified']
    now=datetime.now(timezone.utc)
    elapsed=(now-datetime.fromisoformat(manifest['start_utc'])).total_seconds()/60
    name='ResNet-18 [ft, a, img100k]' if activated else 'ResNet-18 [v, img100k] (original baseline)'
    rows=[table_row('Original ResNet-18','Full validation, rerun here',baseline),
          table_row('Continued ResNet-18','Full validation',status.get('validation')),
          table_row('Original ResNet-18','Held-out test, recorded baseline',historical),
          table_row('Continued ResNet-18','Held-out test, final fixed candidate',status.get('test'))]
    metrics_table='| Model | Split | Images | Accuracy | Balanced accuracy | FAKE precision | FAKE recall | FAKE F1 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    text=f'''# Deadline-bound image accuracy improvement — 21 October 2026

Demo model: **{name}**. Candidate activated: **{activated}**. Handoff prepared after {elapsed:.1f} minutes.

## Historical rollback

The experiment-specific deadline/rollback utilities were retired during final cleanup. Matching backups and manifests remain local; historical scripts can be recovered from Git for inspection. Use the current demo launcher for normal startup. Backup integrity and a real baseline restore/server-health cycle were verified before this historical training run.

## Comparable measured results

{metrics_table}

The original held-out result is the previously recorded FP32 20,000-image evaluation (97.12%), rather than a new baseline test pass. The older outputs/metrics.json reports 97.115% from an earlier kernel/precision path; it is preserved. The prior demo's 96.90% was the separate 2,000-image cohort. Full-validation accuracy is 96.925% for the original checkpoint.

The new checkpoint was selected on complete validation balanced accuracy, with validation loss breaking ties. Threshold stayed 0.5. Train/validation/test folders and FAKE=0, REAL=1 mapping were preserved. The test set was opened once for the final frozen candidate; it did not select an epoch, tune a threshold, or enter training.

## Training decision and provenance

The original ResNet-18 had completed two epochs on 100,000 images, and its validation loss was still improving. Its measured original epoch durations were 297–362 seconds. Cached alternatives were EfficientNet-B0, DeiT-Tiny and ResNet-18; no stronger ResNet/Inception weights were cached. Starting a new backbone or hybrid did not offer a dependable time budget. This run continued the original weights, kept the final residual stage and classifier trainable, retained the original augmentation, and used AdamW at 3e-5 with weight decay 1e-4 and ReduceLROnPlateau. A fresh optimizer was necessary because the original checkpoint did not contain optimizer state.

Completed additional epochs: {len(history)}. Selected global epoch: {selected.get('epoch','none')}. Training used AMP; final evaluation used FP32. Input stayed full-frame RGB, PIL bilinear resize to 224, and ImageNet normalization. No new dependencies or pretrained weights were downloaded.

Baseline checkpoint SHA-256: `{next(item['sha256'] for item in manifest['files'] if item['path']=='models/best_resnet18.pth')}`.

Candidate checkpoint: `{selected.get('checkpoint','none')}`.

Candidate checkpoint SHA-256: `{selected.get('checkpoint_sha256','none')}`.

## Epoch trace (training values are not benchmark results)

| Global epoch | Training accuracy | Full validation accuracy | Validation loss | Seconds |
|---|---:|---:|---:|---:|
'''
    for row in history:
        text+=f"| {row['original_epoch']} | {row['train_accuracy']:.3%} | {row['validation']['accuracy']:.3%} | {row['validation']['loss']:.6f} | {row['seconds']:.1f} |\n"
    text+='\n## Confusion matrices\n\nRows and columns are REAL, FAKE.\n\n'
    for title,metrics in [('Baseline full validation',baseline),('Candidate full validation',status.get('validation')),('Recorded baseline held-out test',historical),('Candidate held-out test',status.get('test'))]:
        if metrics:
            matrix=metrics.get('confusion_matrix_real_fake',metrics.get('confusion_matrix'))
            text+=f'{title}: `{matrix}`.\n\n'
    if deployment:
        text+='## Demo verification\n\n'+json.dumps(deployment['gates'],indent=2)+'\n\n'
        for row in deployment['sanity']:
            text+=f"- {row['input']}: {row['prediction']}, FAKE probability {row['fake_probability']:.3%}; {row['nonconstant_methods']}/6 nonconstant maps; warm prediction {row['warm_prediction_seconds']*1000:.1f} ms; full gallery {row['complete_gallery_seconds']:.2f} s.\n"
    text+='''
## Limits

These are within-dataset results on Real vs Fake Faces and do not establish accuracy on arbitrary internet photos or DFDC video. The same validation split was used for epoch selection, as in the baseline. Heatmaps show model influence rather than verified edit boundaries. The broader video benchmark and its models were not changed. Checkpoints, rollback bundle and per-image evaluation CSVs remain local in this isolated run directory.
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8')
    if activated:
        readme=PROJECT_ROOT/'README.md'
        contents=readme.read_text(encoding='utf-8')
        section=f'''## Demo image continuation — 21 October 2026

The current image demo uses **ResNet-18 [ft, a, img100k]**, selected on full validation and verified through the actual image/XAI UI. Validation accuracy improved from **{baseline['accuracy']:.3%}** to **{status['validation']['accuracy']:.3%}** on the same 20,000 images. The fixed candidate's held-out test accuracy is **{status['test']['accuracy']:.3%}** on 20,000 images, compared with the recorded baseline's **{historical['accuracy']:.3%}**. Full validation/test balanced accuracy, fake-class precision/recall/F1, confusion matrices and checkpoint identities are in [the local executed report]({root.relative_to(PROJECT_ROOT).as_posix()}/REPORT.md).

Activation uses `configs/runtime/active_image_model.json`; original weights are preserved. Historical rollback data remains local; the expired deadline/rollback utilities were retired during final cleanup. The historical benchmark sections below retain their original results.

'''
        marker='## Demo image continuation — 21 October 2026'
        if marker not in contents:
            position=contents.index('\n\n')+2
            readme.write_text(contents[:position]+section+contents[position:],encoding='utf-8')
        guide=PROJECT_ROOT/'docs/automatic_gallery.md'
        guide_text=guide.read_text(encoding='utf-8')
        guide_section=f'''## Current image checkpoint — 21 October 2026

The recommended image model is **ResNet-18 [ft, a, img100k]**, where `ft` means continued fine-tuning of the original checkpoint. Its selector shows **{status['validation']['accuracy']:.3%} full validation accuracy** on 20,000 images. The caption separately reports **{status['test']['accuracy']:.3%} held-out test accuracy** on 20,000 images. Ranking still uses the shared 2,000-image validation cohort; the full validation score is explicitly labeled to distinguish these scopes. Internal ID: `R18_FT_20261021`.

The final checkpoint passed reload, repeated inference, all six CNN explanation methods and actual Streamlit uploads. Epoch selection used validation only. The test set was evaluated once after selection. These are within-dataset image results; heatmaps retain the interpretation limits below. The original image checkpoint remains available, and the video models retain their existing results. See [the local measured report](../{root.relative_to(PROJECT_ROOT).as_posix()}/REPORT.md) for precision, recall, F1, confusion matrices and rollback.

'''
        if '## Current image checkpoint — 21 October 2026' not in guide_text:
            position=guide_text.index('\n\n')+2
            guide.write_text(guide_text[:position]+guide_section+guide_text[position:],encoding='utf-8')
    print(f'Wrote {root/"REPORT.md"}; activated={activated}; elapsed={elapsed:.1f} minutes',flush=True)


def report_current(root):
    """Use the existing image-reporting entry point for both current aliases."""
    import csv
    import subprocess
    from training.image_continuation import IDENTITIES, PROTOCOL, historical_hashes
    from training.demo_accuracy import sha, write
    preflight = load(root/'preflight.json')
    deployment = load(root/'deployment.json')
    if not deployment or not deployment['verified'] or not load(root/'promotion.json'):
        raise ValueError('Generate a current deployment report only after verified promotion')
    unchanged = historical_hashes() == preflight['historical_hashes']
    if not unchanged:
        raise ValueError('Historical evidence changed')
    lines = ['# Current ResNet-18 and EfficientNet-B0 image models', '',
             'Both existing logical identities were updated after validation-only selection. No new experiment IDs were created. Historical image evidence and the frozen DFDC video benchmark remain byte-for-byte unchanged.', '',
             'Dataset: predefined 100,000 training images (50,000 per class), 20,000 validation and 20,000 test images. Legacy logits: FAKE=0, REAL=1. Metrics: FAKE positive. RGB, PIL bilinear 224×224, ImageNet normalization; unchanged mild flip/color/rotation training augmentation.', '',
             'The test split has been used in historical reporting. This is a repeated evaluation after both selections and promotion eligibility were frozen. No test metric selected epochs, learning rates, thresholds, or deployment candidates.', '',
             '## Completed training', '',
             '| Identity | Stage epochs | Selected epoch | Early stopping | Batch / effective | Training seconds | Peak allocated GPU MiB |',
             '|---|---|---:|---|---|---:|---:|']
    summaries, metric_rows = {}, []
    for identity in IDENTITIES:
        folder = root/identity
        training = load(folder/'training_finished.json')
        status = load(folder/'final_status.json')
        benchmark = load(folder/'inference_benchmark.json')
        summaries[identity] = {'training': training, 'evaluation': status, 'benchmark': benchmark}
        lines.append(f"| {identity} | {training['start_epoch']}–{training['end_epoch']} | {training['selected_epoch']} | {training['early_stopping_triggered']} | {training['physical_batch_size']} / {training['effective_batch_size']} | {training['training_seconds']:.3f} | {training['peak_vram_bytes']/1024**2:.2f} |")
        for split in ['validation', 'test']:
            metric = status[split]
            metric_rows.append({'identity': identity, 'split': split, 'samples': metric['samples'],
                                **{key: metric[key] for key in ['accuracy', 'balanced_accuracy', 'precision', 'fake_recall', 'f1', 'macro_f1', 'weighted_f1', 'roc_auc', 'pr_auc', 'loss', 'log_loss', 'brier_score', 'false_positives', 'false_negatives', 'false_positive_rate', 'false_negative_rate', 'specificity', 'sensitivity']},
                                'real_precision': metric['real_precision'], 'real_recall': metric['real_recall'], 'real_f1': metric['real_f1'],
                                'parameters': training['parameters'], 'trainable_parameters': training['trainable_parameters'],
                                'checkpoint_bytes': status['checkpoint_bytes'], 'checkpoint_sha256': status['checkpoint_sha256'],
                                'batch32_ms_per_image': benchmark['batches']['32']['ms_per_image'],
                                'batch32_images_per_second': benchmark['batches']['32']['images_per_second'],
                                'singleton_ms': benchmark['batches']['1']['ms_per_image']})
    lines += ['', 'ResNet global numbering includes the six previously completed epochs; only epochs 7–10 were eligible. EfficientNet stage numbering starts at 1; its three prior 12k-image epochs are separate warm-start provenance. Both original checkpoints lacked resumable optimizer state, so initialization was transparently weight-only. New last-state checkpoints include optimizer, scheduler, scaler, RNG, sampler, stopping state, selection hashes and complete trustworthy history.', '',
              f"AdamW: backbone LR {PROTOCOL['backbone_lr']}, ResNet head LR {PROTOCOL['resnet_classifier_lr']}, EfficientNet head LR {PROTOCOL['classifier_lr']}, weight decay {PROTOCOL['weight_decay']}. ReduceLROnPlateau factor {PROTOCOL['scheduler_factor']}, patience {PROTOCOL['scheduler_patience']}. CrossEntropyLoss, CUDA AMP training, FP32 validation. Maximum epoch 10, loss-based early stopping patience 3, min_delta {PROTOCOL['min_delta']}. Selected checkpoints prioritize validation balanced accuracy, with loss breaking exact practical ties.", '',
              'Training seconds sum actually completed epoch training plus validation; discarded pre-commit throughput-probe work and preflight are excluded. The first incomplete ResNet epoch-7 probe was interrupted to profile data loading, then restarted without recording a completed epoch. Per-epoch history records actual worker counts and loading backend; the equivalent threaded decoder reduces Windows memory pressure after profiling. Peak GPU memory uses torch.cuda.max_memory_allocated. All new epochs process exactly 100,000 training images.', '',
              '## Final FP32 metrics', '',
              '| Model | Split | Accuracy | BA | FAKE recall | FAKE F1 | Macro-F1 | ROC-AUC | PR average precision |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in metric_rows:
        lines.append(f"| {row['identity']} | {row['split']} | {row['accuracy']:.3%} | {row['balanced_accuracy']:.3%} | {row['fake_recall']:.3%} | {row['f1']:.6f} | {row['macro_f1']:.6f} | {row['roc_auc']:.8f} | {row['pr_auc']:.8f} |")
    lines += ['', 'Per-class precision/recall/F1, weighted-F1, cross-entropy/log loss, Brier score, FP/FN, rates, sensitivity/specificity and complete measurements are in `current_metrics.csv` and `current_metrics.json`. Confusion matrices below use rows=actual, columns=predicted, order REAL then FAKE.', '']
    for identity, summary in summaries.items():
        training, status, benchmark = summary['training'], summary['evaluation'], summary['benchmark']
        baseline = load(root/identity/'baseline_validation_fp32.json')
        lines += [f'## {identity}', '',
                  f"Same full 20k validation before → after: accuracy {baseline['accuracy']:.3%} → {status['validation']['accuracy']:.3%}; balanced accuracy {baseline['balanced_accuracy']:.3%} → {status['validation']['balanced_accuracy']:.3%}; FAKE recall {baseline['fake_recall']:.3%} → {status['validation']['fake_recall']:.3%}; macro-F1 {baseline['macro_f1']:.6f} → {status['validation']['macro_f1']:.6f}; ROC-AUC {baseline['roc_auc']:.8f} → {status['validation']['roc_auc']:.8f}.", '',
                  f"Validation matrix: `{status['validation']['confusion_matrix_real_fake']}`. Test matrix: `{status['test']['confusion_matrix_real_fake']}`.", '',
                  f"Selected checkpoint: `{status['checkpoint']}`. SHA-256: `{status['checkpoint_sha256']}`. Actual size: {status['checkpoint_bytes']:,} bytes. Parameters: {training['parameters']:,} total; {training['trainable_parameters']:,} trainable. Latest resumable state: `{root/identity/'last_training_state.pt'}`.", '',
                  f"Measured batch-32 FP32 latency: {benchmark['batches']['32']['ms_per_image']:.6f} ms/image; throughput {benchmark['batches']['32']['images_per_second']:.3f} images/s. Single-image latency: {benchmark['batches']['1']['ms_per_image']:.6f} ms. Timing uses the same first 32 validation images, 10 warmups, 30 repeats and CUDA synchronization, excluding preprocessing/transfer/XAI.", '']
        if identity == IDENTITIES[1]:
            historical = load(PROJECT_ROOT/'results/execution_20261019/E02_efficientnet_b0/summary.json')
            lines += [f"Historical E02: 12,000 training images, 2,000-image validation accuracy {historical['validation']['accuracy']:.2%}, best warm-start epoch {historical['best_epoch']}. That subset result is not directly compared with the new full-validation score. The before/after result above evaluates both checkpoints on the same full 20k validation split.", '']
        else:
            historical = load(PROJECT_ROOT/'results/demo_accuracy_20261021_040741/candidate_test_fp32.json')
            lines += [f"Recorded epoch-6 full test → current frozen full test: accuracy {historical['accuracy']:.3%} → {status['test']['accuracy']:.3%}; FAKE recall {historical['fake_recall']:.3%} → {status['test']['fake_recall']:.3%}; macro-F1 {historical['macro_f1']:.6f} → {status['test']['macro_f1']:.6f}; ROC-AUC {historical['roc_auc']:.8f} → {status['test']['roc_auc']:.8f}. Both use the same 20,000 test images; the later result is repeated historical-split evaluation, not an independent benchmark.", '']
        lines += ['| Epoch | Train loss | Train accuracy | Validation loss | Validation BA | LRs used | Seconds |',
                  '|---:|---:|---:|---:|---:|---|---:|']
        for row in training['history']:
            validation = row['validation']
            ba = f"{validation['balanced_accuracy']:.5%}" if 'balanced_accuracy' in validation else 'not recorded'
            lines.append(f"| {row['epoch']} | {row['train_loss']:.8f} | {row['train_accuracy']:.5%} | {validation['loss']:.8f} | {ba} | {row.get('learning_rates_used', 'historical; see original logs')} | {row['seconds']:.3f} |")
        lines.append('')
    verification = load(root/'verification.json')
    lines += ['## Deployment and verification', '',
              'The atomic `configs/runtime/current_image_models.json` pointer updates both stable IDs; `active_image_model.json` remains compatible with the continued ResNet alias. Each pointer binds a verified deployment summary and checkpoint hashes. The catalog replaces existing IDs instead of appending duplicates. Prediction and explanation caches include checkpoint hashes.', '',
              f"Executed gates: `{json.dumps(verification['gates'], sort_keys=True)}`. Every CNN explanation method was checked on both REAL and FAKE validation inputs for finite 224×224 maps and unchanged model weights/probabilities. Actual Streamlit uploads rendered the automatic six-method galleries for both selected architectures; the video view was checked. IG convergence and LIME fit diagnostics remain visible, rather than being treated as proof of attribution faithfulness.", '',
              f"Historical integrity: {len(preflight['historical_hashes'])} tracked historical files retain their original SHA-256 values. Rollback copies and their hashes are in `{root/'preflight.json'}` and `{root/'rollback'}`. To restore the previous deployment, restore the saved `configs/runtime/active_image_model.json` and remove only the new `configs/runtime/current_image_models.json` pointer. Historical checkpoints were never overwritten.", '',
              'Portable PNG/SVG charts and source hashes are in `docs/figures/current_image_models/`. Regenerate with `python -B -m tools.plot_demo_accuracy --current`; regenerate this report with `python -B -m tools.report_demo_accuracy --current --output results/current_image_models`.', '',
              '## Limits', '',
              'These are within-dataset image results. No identity-disjoint or new external benchmark was added. Model explanations are not ground-truth manipulation masks. GPU-only timing excludes user-upload decoding and attribution. Source checkpoints, prediction evidence, rollback data, and machine-local pointers remain local under repository artifact policy.']
    tests = load(root/'regression.json')
    if tests:
        lines += ['', 'Actual regression execution: `'+json.dumps(tests, sort_keys=True)+'`.']
    modified = subprocess.check_output(['git', 'diff', '--name-only'], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'],
                                        cwd=PROJECT_ROOT, text=True).splitlines()
    source_files = sorted({path for path in modified+untracked if path.endswith('.py')})
    lines += ['', '## Changed source files', '']+[f'- `{path}`' for path in source_files]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (root/'current_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    write(root/'current_metrics.json', {'models': summaries, 'rows': metric_rows, 'protocol': PROTOCOL,
                                       'historical_evidence_unchanged': unchanged, 'deployment_summary_sha256': sha(root/'deployment.json')})
    print(f'Generated current image report and metrics in {root}', flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--activated',action='store_true')
    parser.add_argument('--current',action='store_true')
    args=parser.parse_args()
    if args.current:
        report_current(args.output.resolve())
    else:
        run(args.output,args.activated)
