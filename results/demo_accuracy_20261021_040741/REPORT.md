# Deadline-bound image accuracy improvement — 21 October 2026

Demo model: **ResNet-18 [ft, a, img100k]**. Candidate activated: **True**. Handoff prepared after 47.4 minutes.

## Immediate rollback

```powershell
powershell -ExecutionPolicy Bypass -File "D:\sundarCollege\Deepfake_Detect\Deepfakereserch\rollback_demo.ps1"
```

Rollback stops only this run's registered experiment/verifier processes, restores the exact baseline app, registry, README and ignored image checkpoint, removes the activation pointer, and restarts the local demo. Dataset files, unrelated work, experiment evidence and video checkpoints are preserved. Backup integrity and a real baseline restore/server-health cycle were verified before training.

## Comparable measured results

| Model | Split | Images | Accuracy | Balanced accuracy | FAKE precision | FAKE recall | FAKE F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Original ResNet-18 | Full validation, rerun here | 20,000 | 96.925% | 96.925% | 98.774% | 95.030% | 0.96866 |
| Continued ResNet-18 | Full validation | 20,000 | 98.655% | 98.655% | 98.563% | 98.750% | 0.98656 |
| Original ResNet-18 | Held-out test, recorded baseline | 20,000 | 97.120% | 97.120% | 98.930% | 95.270% | 0.97066 |
| Continued ResNet-18 | Held-out test, final fixed candidate | 20,000 | 98.695% | 98.695% | 98.593% | 98.800% | 0.98696 |

The original held-out result is the previously recorded FP32 20,000-image evaluation (97.12%), rather than a new baseline test pass. The older outputs/metrics.json reports 97.115% from an earlier kernel/precision path; it is preserved. The prior demo's 96.90% was the separate 2,000-image cohort. Full-validation accuracy is 96.925% for the original checkpoint.

The new checkpoint was selected on complete validation balanced accuracy, with validation loss breaking ties. Threshold stayed 0.5. Train/validation/test folders and FAKE=0, REAL=1 mapping were preserved. The test set was opened once for the final frozen candidate; it did not select an epoch, tune a threshold, or enter training.

## Training decision and provenance

The original ResNet-18 had completed two epochs on 100,000 images, and its validation loss was still improving. Its measured original epoch durations were 297–362 seconds. Cached alternatives were EfficientNet-B0, DeiT-Tiny and ResNet-18; no stronger ResNet/Inception weights were cached. Starting a new backbone or hybrid did not offer a dependable time budget. This run continued the original weights, kept the final residual stage and classifier trainable, retained the original augmentation, and used AdamW at 3e-5 with weight decay 1e-4 and ReduceLROnPlateau. A fresh optimizer was necessary because the original checkpoint did not contain optimizer state.

Completed additional epochs: 4. Selected global epoch: 6. Training used AMP; final evaluation used FP32. Input stayed full-frame RGB, PIL bilinear resize to 224, and ImageNet normalization. No new dependencies or pretrained weights were downloaded.

Baseline checkpoint SHA-256: `406c11e8ddac13a1364483734d1bcbc829add2e367b816bc40b6e9ed44121d06`.

Candidate checkpoint: `D:\sundarCollege\Deepfake_Detect\Deepfakereserch\results\demo_accuracy_20261021_040741\selected_final.pth`.

Candidate checkpoint SHA-256: `314f1394a1876cd6b41d3ddf80b09aef98f6960df31a58f9dc8b73f46319b0a0`.

## Epoch trace (training values are not benchmark results)

| Global epoch | Training accuracy | Full validation accuracy | Validation loss | Seconds |
|---|---:|---:|---:|---:|
| 3 | 99.495% | 98.240% | 0.048074 | 495.2 |
| 4 | 99.835% | 98.500% | 0.043745 | 423.2 |
| 5 | 98.955% | 98.555% | 0.039798 | 415.1 |
| 6 | 99.333% | 98.655% | 0.038194 | 409.6 |

## Confusion matrices

Rows and columns are REAL, FAKE.

Baseline full validation: `[[9882, 118], [497, 9503]]`.

Candidate full validation: `[[9856, 144], [125, 9875]]`.

Recorded baseline held-out test: `[[9897, 103], [473, 9527]]`.

Candidate held-out test: `[[9859, 141], [120, 9880]]`.

## Demo verification

{
  "full_validation_improved": true,
  "common_validation_improved": true,
  "reload": true,
  "inference": true,
  "xai": true,
  "ui": true,
  "latency": true,
  "rollback_roundtrip": true,
  "baseline_checkpoints_preserved": true
}

- random-person.jpeg: FAKE, FAKE probability 93.904%; 6/6 nonconstant maps; warm prediction 69.3 ms; full gallery 1.63 s.
- real.jpeg: REAL, FAKE probability 0.004%; 6/6 nonconstant maps; warm prediction 52.9 ms; full gallery 0.88 s.

## Limits

These are within-dataset results on Real vs Fake Faces and do not establish accuracy on arbitrary internet photos or DFDC video. The same validation split was used for epoch selection, as in the baseline. Heatmaps show model influence rather than verified edit boundaries. The broader video benchmark and its models were not changed. Checkpoints, rollback bundle and per-image evaluation CSVs remain local in this isolated run directory.

## Final active handoff

The active, unpatched UI was verified to select `R18_FT_20261021`, display 98.655% full validation accuracy and 98.695% held-out test accuracy, and retain S6 as the video default. The localhost server health check passed after the final restart. Both focused activation-integrity tests passed. All 62 baseline checkpoint hashes were rechecked after rollback and activation.
