# Current ResNet-18 and EfficientNet-B0 image models

Both existing logical identities were updated after validation-only selection. No new experiment IDs were created. Historical image evidence and the frozen DFDC video benchmark remain byte-for-byte unchanged.

Dataset: predefined 100,000 training images (50,000 per class), 20,000 validation and 20,000 test images. Legacy logits: FAKE=0, REAL=1. Metrics: FAKE positive. RGB, PIL bilinear 224×224, ImageNet normalization; unchanged mild flip/color/rotation training augmentation.

The test split has been used in historical reporting. This is a repeated evaluation after both selections and promotion eligibility were frozen. No test metric selected epochs, learning rates, thresholds, or deployment candidates.

## Completed training

| Identity | Stage epochs | Selected epoch | Early stopping | Batch / effective | Training seconds | Peak allocated GPU MiB |
|---|---|---:|---|---|---:|---:|
| R18_FT_20261021 | 7–10 | 9 | False | 64 / 64 | 2543.745 | 662.43 |
| E02_efficientnet_b0 | 1–10 | 10 | False | 64 / 64 | 4820.199 | 782.46 |

ResNet global numbering includes the six previously completed epochs; only epochs 7–10 were eligible. EfficientNet stage numbering starts at 1; its three prior 12k-image epochs are separate warm-start provenance. Both original checkpoints lacked resumable optimizer state, so initialization was transparently weight-only. New last-state checkpoints include optimizer, scheduler, scaler, RNG, sampler, stopping state, selection hashes and complete trustworthy history.

AdamW: backbone LR 3e-05, ResNet head LR 3e-05, EfficientNet head LR 0.0001, weight decay 0.0001. ReduceLROnPlateau factor 0.5, patience 1. CrossEntropyLoss, CUDA AMP training, FP32 validation. Maximum epoch 10, loss-based early stopping patience 3, min_delta 0.0001. Selected checkpoints prioritize validation balanced accuracy, with loss breaking exact practical ties.

Training seconds sum actually completed epoch training plus validation; discarded pre-commit throughput-probe work and preflight are excluded. The first incomplete ResNet epoch-7 probe was interrupted to profile data loading, then restarted without recording a completed epoch. Per-epoch history records actual worker counts and loading backend; the equivalent threaded decoder reduces Windows memory pressure after profiling. Peak GPU memory uses torch.cuda.max_memory_allocated. All new epochs process exactly 100,000 training images.

## Final FP32 metrics

| Model | Split | Accuracy | BA | FAKE recall | FAKE F1 | Macro-F1 | ROC-AUC | PR average precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R18_FT_20261021 | validation | 98.915% | 98.915% | 98.720% | 0.989129 | 0.989150 | 0.99935786 | 0.99935523 |
| R18_FT_20261021 | test | 98.905% | 98.905% | 98.640% | 0.989021 | 0.989050 | 0.99939991 | 0.99941661 |
| E02_efficientnet_b0 | validation | 98.120% | 98.120% | 98.770% | 0.981321 | 0.981199 | 0.99854631 | 0.99856980 |
| E02_efficientnet_b0 | test | 98.060% | 98.060% | 98.620% | 0.980708 | 0.980599 | 0.99841728 | 0.99844506 |

Per-class precision/recall/F1, weighted-F1, cross-entropy/log loss, Brier score, FP/FN, rates, sensitivity/specificity and complete measurements are in `current_metrics.csv` and `current_metrics.json`. Confusion matrices below use rows=actual, columns=predicted, order REAL then FAKE.

## R18_FT_20261021

Same full 20k validation before → after: accuracy 98.655% → 98.915%; balanced accuracy 98.655% → 98.915%; FAKE recall 98.750% → 98.720%; macro-F1 0.986550 → 0.989150; ROC-AUC 0.99914448 → 0.99935786.

Validation matrix: `[[9911, 89], [128, 9872]]`. Test matrix: `[[9917, 83], [136, 9864]]`.

Selected checkpoint: `D:\sundarCollege\Deepfake_Detect\Deepfakereserch\results\current_image_models\R18_FT_20261021\selected_final.pth`. SHA-256: `55ad833b1c75aa58e2aaafa4609a3b5ed98e52439b795c775a1e9f5323bf89ad`. Actual size: 44,804,363 bytes. Parameters: 11,177,538 total; 8,394,754 trainable. Latest resumable state: `D:\sundarCollege\Deepfake_Detect\Deepfakereserch\results\current_image_models\R18_FT_20261021\last_training_state.pt`.

Measured batch-32 FP32 latency: 0.629489 ms/image; throughput 1588.591 images/s. Single-image latency: 3.983637 ms. Timing uses the same first 32 validation images, 10 warmups, 30 repeats and CUDA synchronization, excluding preprocessing/transfer/XAI.

Recorded epoch-6 full test → current frozen full test: accuracy 98.695% → 98.905%; FAKE recall 98.800% → 98.640%; macro-F1 0.986950 → 0.989050; ROC-AUC 0.99927120 → 0.99939991. Both use the same 20,000 test images; the later result is repeated historical-split evaluation, not an independent benchmark.

| Epoch | Train loss | Train accuracy | Validation loss | Validation BA | LRs used | Seconds |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.18054040 | 92.56100% | 0.09219816 | not recorded | historical; see original logs | 362.067 |
| 2 | 0.07252407 | 97.21900% | 0.08206658 | not recorded | historical; see original logs | 296.561 |
| 3 | 0.01705586 | 99.49500% | 0.04807436 | 98.24000% | historical; see original logs | 495.228 |
| 4 | 0.00648299 | 99.83500% | 0.04374513 | 98.50000% | historical; see original logs | 423.248 |
| 5 | 0.02861301 | 98.95500% | 0.03979789 | 98.55500% | historical; see original logs | 415.110 |
| 6 | 0.01822552 | 99.33300% | 0.03819373 | 98.65500% | historical; see original logs | 409.599 |
| 7 | 0.02197130 | 99.19900% | 0.03799985 | 98.68000% | [3e-05] | 984.472 |
| 8 | 0.02085933 | 99.23000% | 0.03777597 | 98.69500% | [3e-05] | 540.952 |
| 9 | 0.01838710 | 99.31000% | 0.03370141 | 98.91500% | [3e-05] | 509.273 |
| 10 | 0.01441698 | 99.49400% | 0.03823078 | 98.70000% | [3e-05] | 509.047 |

## E02_efficientnet_b0

Same full 20k validation before → after: accuracy 91.655% → 98.120%; balanced accuracy 91.655% → 98.120%; FAKE recall 94.240% → 98.770%; macro-F1 0.916494 → 0.981199; ROC-AUC 0.97494676 → 0.99854631.

Validation matrix: `[[9747, 253], [123, 9877]]`. Test matrix: `[[9750, 250], [138, 9862]]`.

Selected checkpoint: `D:\sundarCollege\Deepfake_Detect\Deepfakereserch\results\current_image_models\E02_efficientnet_b0\selected_final.pth`. SHA-256: `2389a6c08ebcec818aefe22572a23e040056251e008724cc45e6ac2d030bf77e`. Actual size: 16,341,487 bytes. Parameters: 4,010,110 total; 1,131,954 trainable. Latest resumable state: `D:\sundarCollege\Deepfake_Detect\Deepfakereserch\results\current_image_models\E02_efficientnet_b0\last_training_state.pt`.

Measured batch-32 FP32 latency: 0.938023 ms/image; throughput 1066.072 images/s. Single-image latency: 14.472900 ms. Timing uses the same first 32 validation images, 10 warmups, 30 repeats and CUDA synchronization, excluding preprocessing/transfer/XAI.

Historical E02: 12,000 training images, 2,000-image validation accuracy 92.45%, best warm-start epoch 3. That subset result is not directly compared with the new full-validation score. The before/after result above evaluates both checkpoints on the same full 20k validation split.

| Epoch | Train loss | Train accuracy | Validation loss | Validation BA | LRs used | Seconds |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.20704280 | 91.55300% | 0.14017387 | 94.64500% | [0.0001, 3e-05] | 521.478 |
| 2 | 0.15356080 | 94.00600% | 0.10930585 | 95.93500% | [0.0001, 3e-05] | 534.355 |
| 3 | 0.12334274 | 95.29800% | 0.08802614 | 96.73500% | [0.0001, 3e-05] | 524.325 |
| 4 | 0.10219917 | 96.11600% | 0.07681912 | 97.10000% | [0.0001, 3e-05] | 528.382 |
| 5 | 0.08563379 | 96.83900% | 0.06962847 | 97.31000% | [0.0001, 3e-05] | 518.910 |
| 6 | 0.07320960 | 97.25000% | 0.06471937 | 97.54000% | [0.0001, 3e-05] | 461.252 |
| 7 | 0.06372166 | 97.66200% | 0.06134541 | 97.62000% | [0.0001, 3e-05] | 449.768 |
| 8 | 0.05654681 | 97.89800% | 0.05305247 | 98.01000% | [0.0001, 3e-05] | 434.630 |
| 9 | 0.04977115 | 98.20900% | 0.05205532 | 98.10500% | [0.0001, 3e-05] | 425.341 |
| 10 | 0.04342541 | 98.44200% | 0.04972190 | 98.12000% | [0.0001, 3e-05] | 421.758 |

## Deployment and verification

The atomic `configs/runtime/current_image_models.json` pointer updates both stable IDs; `active_image_model.json` remains compatible with the continued ResNet alias. Each pointer binds a verified deployment summary and checkpoint hashes. The catalog replaces existing IDs instead of appending duplicates. Prediction and explanation caches include checkpoint hashes.

Executed gates: `{"historical_evidence_unchanged": true, "preprocessing_and_labels": true, "reload": true, "streamlit_uploads": true, "unique_existing_ids": true, "video_view": true, "xai_finite_aligned": true, "xai_preserves_weights_predictions": true}`. Every CNN explanation method was checked on both REAL and FAKE validation inputs for finite 224×224 maps and unchanged model weights/probabilities. Actual Streamlit uploads rendered the automatic six-method galleries for both selected architectures; the video view was checked. IG convergence and LIME fit diagnostics remain visible, rather than being treated as proof of attribution faithfulness.

Historical integrity: 169 tracked historical files retain their original SHA-256 values. Rollback copies and their hashes are in `D:\sundarCollege\Deepfake_Detect\Deepfakereserch\results\current_image_models\preflight.json` and `D:\sundarCollege\Deepfake_Detect\Deepfakereserch\results\current_image_models\rollback`. To restore the previous deployment, restore the saved `configs/runtime/active_image_model.json` and remove only the new `configs/runtime/current_image_models.json` pointer. Historical checkpoints were never overwritten.

Portable PNG/SVG charts and source hashes are in `docs/figures/current_image_models/`. Regenerate with `python -B -m tools.plot_demo_accuracy --current`; regenerate this report with `python -B -m tools.report_demo_accuracy --current --output results/current_image_models`.

## Limits

These are within-dataset image results. No identity-disjoint or new external benchmark was added. Model explanations are not ground-truth manipulation masks. GPU-only timing excludes user-upload decoding and attribution. Source checkpoints, prediction evidence, rollback data, and machine-local pointers remain local under repository artifact policy.

Actual regression execution: `{"command": "dfenv/Scripts/python.exe -B -m unittest discover -s tests -v", "exit_code": 0, "log": "results/current_image_models/regression_final.log", "passed": true, "seconds": 65.571, "tests_run": 70}`.

## Changed source files

- `app.py`
- `inference/catalog.py`
- `inference/demo_candidate.py`
- `tests/test_demo_candidate.py`
- `tests/test_image_continuation.py`
- `tests/test_parallel_explanations.py`
- `tools/plot_demo_accuracy.py`
- `tools/report_demo_accuracy.py`
- `tools/report_unified.py`
- `tools/verify_demo_accuracy.py`
- `training/demo_accuracy.py`
- `training/image_continuation.py`
