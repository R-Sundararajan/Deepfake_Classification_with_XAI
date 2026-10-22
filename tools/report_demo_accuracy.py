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
    rollback=f'powershell -ExecutionPolicy Bypass -File "{PROJECT_ROOT / "rollback_demo.ps1"}"'
    rows=[table_row('Original ResNet-18','Full validation, rerun here',baseline),
          table_row('Continued ResNet-18','Full validation',status.get('validation')),
          table_row('Original ResNet-18','Held-out test, recorded baseline',historical),
          table_row('Continued ResNet-18','Held-out test, final fixed candidate',status.get('test'))]
    metrics_table='| Model | Split | Images | Accuracy | Balanced accuracy | FAKE precision | FAKE recall | FAKE F1 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    text=f'''# Deadline-bound image accuracy improvement — 21 October 2026

Demo model: **{name}**. Candidate activated: **{activated}**. Handoff prepared after {elapsed:.1f} minutes.

## Immediate rollback

```powershell
{rollback}
```

Rollback stops only this run's registered experiment/verifier processes, restores the exact baseline app, registry, README and ignored image checkpoint, removes the activation pointer, and restarts the local demo. Dataset files, unrelated work, experiment evidence and video checkpoints are preserved. Backup integrity and a real baseline restore/server-health cycle were verified before training.

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

Activation uses `configs/runtime/active_image_model.json`; original weights are preserved. Restore the original demo with `powershell -ExecutionPolicy Bypass -File "{PROJECT_ROOT / 'rollback_demo.ps1'}"`. The experimental checkpoint and rollback bundle are local artifacts. The historical benchmark sections below retain their original results.

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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--activated',action='store_true')
    args=parser.parse_args()
    run(args.output,args.activated)
