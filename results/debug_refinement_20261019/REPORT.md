# Executed debugging and demo refinement — 19 October 2026

## Verified causes and corrections

* **Training-domain transfer and preprocessing sensitivity, not an inverted label mapping.** S2 reproduces 16.1258% FAKE on `random-person.jpeg`. On the common 2,000-image validation cohort its native DFDC-aligned transfer reaches only 51.10% balanced accuracy. It still reaches 81.40% on its video validation protocol. Both mappings (legacy FAKE=0 and unified FAKE=1) are explicitly regression-tested. Checkpoint, encoder hashes and preprocessing contracts are checked before loading.
* **The reported real-photo example was S5.** S5 reproduces 66.1481% REAL / 33.8519% FAKE. S2 instead gives that file 64.4514% FAKE with its aligned crop. All model/input pairs are retained below; these user-supplied labels are sanity checks, not an independent benchmark.
* **Preprocessing is model specific.** Applying the same DFDC-aligned synthetic crop to S0 changes its FAKE score from 85.7806% to 0.0304%; S0 therefore does not outperform S2 under every identical preprocessing choice. Under full-frame preprocessing S0 correctly labels both supplied inputs; S2 still misses the synthetic face (24.1582% FAKE) but gets the real photo right (2.4649% FAKE). Neither architecture nor a universal face-crop policy alone explains the difference. Native S2 alignment matches its training/evaluation pipeline and was retained. Actual affine matrices, original-coordinate crop corners, detector results, normalized tensors and crop PNGs are saved in `corrected/diagnostics.json` and its model directories.
* **An actual legacy CLI interpolation defect was repaired** in [inference/predict.py](../../inference/predict.py): implicit Pillow resizing used bicubic while the evaluation/Streamlit path uses bilinear. The legacy CLI synthetic score changes from 94.2078% to 85.8181% FAKE after matching interpolation; small CUDA memory-layout/kernel differences remain relative to the unified path's 85.7806%. RGB conversion, shape, pixel range and ImageNet normalization match. The image-trained checkpoints intentionally use full frames; aligned DFDC heads retain their native alignment. See `preprocessing/checks.json`.
* **IG discarded sign and used insufficient integration steps.** The old S2 synthetic REAL explanation had completeness residual 0.838731; old S0 synthetic residual was 4.440251. [explainability/methods.py](../../explainability/methods.py) now retains signed channel sums, raw positive/negative statistics and magnitude quantiles, batches Gauss–Legendre integration, and reports adaptive convergence. Default budgets are 32–128 steps, with 256 selectable. The S2 synthetic mean-reference residual falls to 0.002813 at 256 steps; that cold/adaptive call took 12.81 s, so it is not the default. A blurred-input reference converged at 32 steps in 0.515 s for that case. The blurred reference explains changes from the blurred image, a different attribution question; it is labeled explicitly.
* **The old overlay tinted zero activation blue.** [explainability/gradcam.py](../../explainability/gradcam.py) now uses activation-dependent opacity, so a zero map leaves the image unchanged. Signed maps use blue/white/red for opposing/zero/supporting contributions. The old Grad-CAM++ absolute squared-gradient cutoff also discarded small valid gradients; a dedicated scaled-score regression now covers this case. The method remains labeled a first-gradient-power approximation.
* **Interaction repeated inference and XAI unnecessarily.** [app.py](../../app.py) now caches two resident predictors, four deterministic predictions and eight completed explanations. Keys bind checkpoint, preserved input path/content hash, modality, target, method, parameters and implementation version. Explanations run only after pressing Generate; changing opacity reuses attribution. Models share a lock to protect mutable detector/hook state. Expensive LIME is never generated automatically. Cached checksum verification also avoids hashing unchanged checkpoints on every interaction.
* **Video explanations preserve the selected decision.** [inference/unified.py](../../inference/unified.py) expands fixed frame context for perturbation/IG batches, differentiates the selected complete spatial/temporal/fusion system, and includes video temperature for probability-based explanations. Native differentiable LSTM evaluation avoids the cuDNN eval-backward restriction. Image fallbacks remain explicitly spatial; their caveats are in the collapsed bottom section.

All 38 before/after image prediction pairs preserve the deployed FAKE probability exactly (maximum change 0); UI/XAI repairs do not constitute an accuracy gain for the existing frozen models. The CLI interpolation correction is reported separately.

## All supplied-image results

Expected synthetic label: FAKE. Expected photograph label: REAL. Both FAKE and REAL probabilities are shown. S6/S7 image runs use S2; S8 uses S5. Smoke artifacts are included for completeness but excluded from all leaderboards. Every native crop, tensor, face-detection result, geometry, checkpoint hash, threshold, calibration and inference/explanation timing is retained in [diagnostics.json](corrected/diagnostics.json). Source snapshots before the repairs are under `baseline/source_snapshot`.

| Checkpoint | Training domain | Synthetic prediction | Synthetic FAKE / REAL | Photo prediction | Photo FAKE / REAL |
|---|---|---|---|---|---|
| S0 | Real vs Fake Faces | FAKE | 85.78% / 14.22% | REAL | 0.37% / 99.63% |
| S1 | Local source-grouped DFDC | FAKE | 85.95% / 14.05% | FAKE | 93.13% / 6.87% |
| S1p | Local source-grouped DFDC | FAKE | 99.94% / 0.06% | FAKE | 99.99% / 0.01% |
| S1b | Local source-grouped DFDC | REAL | 16.25% / 83.75% | FAKE | 64.92% / 35.08% |
| S2 | Local source-grouped DFDC | REAL | 16.13% / 83.87% | FAKE | 64.45% / 35.55% |
| S3 | Local source-grouped DFDC | REAL | 42.20% / 57.80% | REAL | 43.73% / 56.27% |
| S4 | Local source-grouped DFDC | FAKE | 93.57% / 6.43% | FAKE | 91.84% / 8.16% |
| S5 | Local source-grouped DFDC | REAL | 35.74% / 64.26% | REAL | 33.85% / 66.15% |
| S6 | Local source-grouped DFDC | REAL | 16.13% / 83.87% | FAKE | 64.45% / 35.55% |
| S7 | Local source-grouped DFDC | REAL | 16.13% / 83.87% | FAKE | 64.45% / 35.55% |
| S8 | Local source-grouped DFDC | REAL | 35.74% / 64.26% | REAL | 33.85% / 66.15% |
| E01_resnet18 | Real vs Fake Faces | FAKE | 99.33% / 0.67% | REAL | 0.01% / 99.99% |
| E02_efficientnet_b0 | Real vs Fake Faces | FAKE | 99.31% / 0.69% | REAL | 0.30% / 99.70% |
| E03_attention | Real vs Fake Faces | FAKE | 99.04% / 0.96% | REAL | 0.23% / 99.77% |
| E04_jpeg | Real vs Fake Faces | FAKE | 99.23% / 0.77% | REAL | 0.05% / 99.95% |
| smoke_E01_resnet18 | Real vs Fake Faces | REAL | 38.46% / 61.54% | REAL | 14.83% / 85.17% |
| smoke_E02_efficientnet_b0 | Real vs Fake Faces | FAKE | 57.68% / 42.32% | REAL | 49.96% / 50.04% |
| smoke_E03_attention | Real vs Fake Faces | FAKE | 51.40% / 48.60% | REAL | 48.12% / 51.88% |
| smoke_E04_jpeg | Real vs Fake Faces | FAKE | 56.92% / 43.08% | FAKE | 50.16% / 49.84% |

## Evaluation and threshold audit

Source groups and original/derivative edges were rechecked against the manifest, including missing-original aliases and exact SHA-256 duplicates: no cross-split source/hash leakage found. Train: 813 videos (52 REAL, 761 FAKE), 54 groups. Validation: 175 (11 REAL, 164 FAKE), 17 groups. Test: 174 (11 REAL, 163 FAKE), 15 groups. The unlabeled video remains quarantined. Actor disjointness is not established.

S2 validation confusion matrix (rows/columns REAL, FAKE) is `[[11,0],[61,103]]`. REAL recall=100%, FAKE recall=62.8049%, REAL precision=15.2778%, FAKE precision=100%, accuracy=65.1429%, balanced accuracy=81.4024%, macro-F1=0.518298. The 61 false REAL predictions explain the low REAL precision/F1. Metrics were recomputed from saved raw validation predictions and matched every saved summary.

Validation-only S2 threshold search finds 0.499221, yielding 81.7073% in-sample balanced accuracy, a gain of only 0.3049 percentage points. Leave-one-source-group-out threshold fitting yields 76.8570% BA on held-out group decisions. This is not stable evidence for deployment, so frozen thresholds remain 0.5. Calibration is bound to checkpoint and modality; images do not inherit video temperatures. No test media/features were reopened for model selection or retraining. [Full audit](evaluation_audit.json).

## Corrected model-family leaderboards

Families sort by their best compatible validation balanced accuracy, then macro-F1, then ROC-AUC. Means include only completed runs on the same validation manifest/unit. Training budgets and preprocessing remain visible. The new separately saved S2 image-transfer benchmark is included at its measured 51.10%, rather than displaying its video accuracy as image accuracy. It does not change video records. Its saved image probabilities are in `domain_pilot/S2_image_validation_predictions.csv`.

### Image validation: common 2,000-image Real vs Fake Faces subset

| Family | Best variant | Best validation BA | Mean validation BA | Comparable runs |
|---|---|---|---|---|
| ResNet / CNN | S0 | 96.90% | 93.85% | 2 |
| EfficientNet | E02_efficientnet_b0 | 92.45% | 81.96% | 4 |

S0 used 100,000 training images; E01–E04 used 12,000. S2 trained on DFDC videos. These are common-evaluation comparisons with different training budgets/domains. The mixed-domain pilot is research only and excluded from production family means. Unbenchmarked transfers can be enabled explicitly, and never receive a video-metric image ranking.

### Video validation: common source-grouped DFDC normalized-eight-frame protocol

| Family | Best variant | Best validation BA | Mean validation BA | Comparable runs |
|---|---|---|---|---|
| Spatial + Temporal Hybrid | S6 | 94.82% | 94.61% | 3 |
| EfficientNet | S1b | 81.40% | 74.07% | 4 |
| Vision Transformer | S4 | 78.69% | 70.51% | 2 |
| CNN + ViT Hybrid | S5 | 72.31% | 72.31% | 1 |
| ResNet / CNN | S0 | 50.00% | 50.00% | 1 |

## XAI implementation and validation

CNNs support Grad-CAM, Grad-CAM++ (first-gradient-power approximation), Layer-CAM, signed IG, Occlusion Sensitivity and LIME. Transformers support IG, Occlusion, LIME, native attention rollout and gradient-weighted attention. Hybrids expose both compatible sets. Every method has a plain-language description; the UI supports REAL/FAKE targets, color legends and opacity. LIME uses the installed `lime.lime_base.LimeBase` weighted local surrogate over a deterministic 7×7 region grid, not claimed semantic objects. Rollout remains class agnostic; gradient attention is class dependent but neither is asserted causal.

Actual checkpoints: 15; image/method/target cases: 368. All maps were finite and 224×224; overlays align with the actual model input; prediction scores were unchanged after explanation. Hook cleanup and target behavior also have analytic-model regressions. 18 of 60 default-budget IG cases did not meet tolerance; 2 maps were constant; 18 LIME fits had weighted R² below 0.5. These cases are flagged instead of being presented as reliable localization. 63/184 predicted-target deletion tests had negative salient-over-random advantage. Mean masks are out of distribution; these are recorded limitations, not a claim of universal attribution faithfulness. Maximum FP32-XAI versus AMP-inference probability difference on these images was 0.004678; no weights are changed for explanations.

[Raw XAI checks](images/checks.json) include raw attribution statistics, signed sums, gradient availability, convergence history, LIME fit, runtimes and perturbation results. PNGs and numeric map arrays are grouped by model/input/method/target. Implementation references: [Captum IG and completeness](https://captum.ai/docs/extension/integrated_gradients), [Layer-CAM reference implementation](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/layer_cam.py), [LIME image perturbation/surrogate implementation](https://github.com/marcotcr/lime/blob/master/lime/lime_image.py).

## Measured latency

Same process/device, synchronized CUDA, resident models, two warmups and seven repeats; medians shown. The original implementation was loaded from saved source snapshots for the before measurements. Checkpoint load, crop/input copy, resize, normalization, GPU transfer, forward, decision and attribution/blending timings are retained separately in [profile.json](profile.json). These numbers exclude browser paint and file upload. First-use GPU startup can still take seconds.

| Model / image | Before incl. Grad-CAM ms | After incl. Grad-CAM ms | After prediction only ms |
|---|---|---|---|
| S0 / random-person.jpeg | 67.71 | 58.62 | 37.39 |
| S0 / real.jpeg | 69.45 | 62.16 | 42.43 |
| S2 / random-person.jpeg | 146.95 | 126.79 | 87.02 |
| S2 / real.jpeg | 130.24 | 112.00 | 69.31 |

Streamlit prediction cache medians: S0 0.989 ms, S2 0.890 ms. Completed-explanation cache medians: S0 0.685 ms, S2 0.709 ms. Actual widget rerender/serialization timings and feature/XAI call counters are saved in [app QA](app_qa_final.json). AppTest creates script threads and includes startup/serialization; its cold video upload measurements are not interchangeable with the resident direct-inference timings below.

## Video QA

The first validation filename alphabetically for each label was selected before inference, rather than choosing clips with correct predictions. All 22 model/video runs reproduced their saved validation probabilities within 0.0002, sampled eight distinct deterministic positions and produced `8×3×224×224` model inputs. The corresponding explanation view reproduces the complete selected calibrated decision within the declared FP32/AMP tolerance. S6 was repeated for deterministic equality and all six CNN methods were tested on its complete video decision. Native attention is a branch visualization with an explicit limitation.

| Model | True label | Prediction | FAKE probability | Resident/cold QA ms |
|---|---|---|---|---|
| S6 | REAL | REAL | 2.4518% | 10121 |
| S6 | FAKE | FAKE | 98.9213% | 1333 |
| S8 | REAL | REAL | 3.7247% | 1421 |
| S8 | FAKE | FAKE | 96.4603% | 1309 |
| S7 | REAL | REAL | 3.5710% | 1343 |
| S7 | FAKE | FAKE | 95.3207% | 1198 |
| S1b | REAL | REAL | 43.3375% | 1203 |
| S1b | FAKE | FAKE | 54.1824% | 1257 |
| S2 | REAL | REAL | 42.3596% | 1241 |
| S2 | FAKE | FAKE | 55.3671% | 1222 |
| S4 | REAL | REAL | 29.6824% | 1214 |
| S4 | FAKE | FAKE | 77.2921% | 1158 |
| S5 | REAL | REAL | 32.5821% | 1336 |
| S5 | FAKE | FAKE | 67.3382% | 1211 |
| S1p | REAL | REAL | 30.4899% | 1214 |
| S1p | FAKE | FAKE | 99.4396% | 1243 |
| S3 | REAL | REAL | 44.9739% | 1275 |
| S3 | FAKE | FAKE | 78.9909% | 1197 |
| S1 | REAL | REAL | 45.0882% | 1285 |
| S1 | FAKE | FAKE | 71.5823% | 1136 |
| S0 | REAL | REAL | 30.9496% | 1376 |
| S0 | FAKE | REAL | 33.0508% | 1274 |

S6 correctly classifies the selected REAL and FAKE clips. S0 misses the selected FAKE clip; this remains visible. MOV support was verified using a **lossless remux**, with S6 REAL probability 97.5482%. Corrupt and missing clips were rejected cleanly. The UI supplies playback, frame previews, per-frame crop status, spatial probability traces, representative-frame XAI and separate temporal sensitivity. Frame traces are not temporal-anomaly proof. Image-only E01–E04 are excluded from the video selector. See [video checks](video/checks.json).

## Limited domain-aware experiment

Registration was saved before feature extraction: existing frozen EfficientNet encoder, original S2 initialization/statistics, 2,000 training images (1,000 per class), 813 DFDC training videos, independent class-balanced replacement sampling within domains, equally weighted domain losses, 16 epochs of 26 steps and LR 0.0005. A matched DFDC-only continuation uses the same video draws, initial head and optimization budget. Image and video validation remain separate; selection uses their harmonic-mean balanced accuracy. No test pixels/features or supplied sanity images enter training, calibration or selection. Extraction took 56.8 s; each tiny-head continuation took approximately 14.9–15.3 s.

| Run | Image validation BA | Image macro-F1 | Video validation BA | Video macro-F1 |
|---|---|---|---|---|
| baseline_S2 | 51.10% | 0.511 | 81.40% | 0.518 |
| dfdc_only_control | 53.00% | 0.528 | 79.57% | 0.495 |
| mixed_domains | 73.35% | 0.733 | 82.32% | 0.530 |

The mixed candidate improves image transfer by 22.25 percentage points over S2 and video BA by 0.9146 points on these validation cohorts. It remains below the existing image/video champions, fails the preregistered deployment rule and is **not deployed**. This is a bounded validation pilot, not new held-out test evidence or proof of broad generalization. Per-class recalls, confusion matrices, calibration, selected weights, epoch histories, probability CSVs, manifests and checkpoint hashes remain in [domain_pilot](domain_pilot/comparison.json). Calibration contexts are separately bound to their unit/checkpoint in `domain_pilot/calibration_context.json`.

## Regression, UI and remaining limitations

56 regression tests: passed; 19 actual Streamlit widget/upload/cache checks passed, including target changes, opacity reuse, grouped selection, explicit S8 image fallback, REAL/FAKE videos and corrupt input. Notes/warnings stay collapsed at the page bottom on initial, result and failure paths. The live server was restarted to remove stale imports and verified at `http://127.0.0.1:8501`.

Small validation REAL support (11 videos), source grouping without actor guarantees, the shared calibration/selection set, unvalidated selective-risk thresholds, domain-sensitive crops, incomplete video coverage, weak/inconclusive attribution cases and cold startup costs remain. No semantic facial artifact or forensic certainty is claimed. No historical checkpoints, frozen test predictions or XLSX research workbooks were rewritten.

## Reproduce

Use the existing `dfenv` from the project root. Commands that generate diagnostics require new output directories; the recorded directories are intentionally protected.

```powershell
.\dfenv\Scripts\python.exe -m unittest discover -s tests -v
.\dfenv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
.\dfenv\Scripts\python.exe -m tools.diagnose_refinement --output results/refinement_rerun/inputs
.\dfenv\Scripts\python.exe -m tools.audit_refinement --output results/refinement_rerun/audit
.\dfenv\Scripts\python.exe -m tools.verify_refinement images --output results/refinement_rerun/images
.\dfenv\Scripts\python.exe -m tools.verify_refinement video --output results/refinement_rerun/video
.\dfenv\Scripts\python.exe -m tools.verify_app_refinement --output results/refinement_rerun/app_qa.json
.\dfenv\Scripts\python.exe -m tools.profile_refinement --output results/refinement_rerun/profile
.\dfenv\Scripts\python.exe -m training.domain_pilot --output results/refinement_rerun/domain_pilot --epochs 16 --per-class 1000 --domain-weight 0.5
.\dfenv\Scripts\python.exe -m tools.report_refinement
```

On restricted Windows runners, Streamlit AppTest needs local loopback/socket access for its event loop. The other commands do not require a public service or new dataset download. Report regeneration reads existing saved results and updates derived Markdown only.

## Supplementary video-method coverage

38 additional method/video cases passed across S8 (all 8 methods), S4 (all 5) and S0 (all 6), on both labeled validation clips. Maps were finite and predictions unchanged. Together with the S6 checks this covers complete spatial, temporal, transformer and combined decision paths. [Supplementary checks](video/additional_methods.json). Repeat these architecture-level checks by adding `--all-methods` to the documented `tools.verify_refinement video` command, using a new output folder.

## Post-selection pilot sanity images

These were run only after validation selection and the decision not to deploy. They did not enter training, calibration, thresholds or deployment selection. The mixed head gets the synthetic sanity image right but still misclassifies the real photo. This reinforces the remaining generalization limitation.

| Candidate | Input | User label | Prediction | FAKE / REAL |
|---|---|---|---|---|
| dfdc_only_control | random-person.jpeg | FAKE | REAL | 23.73% / 76.27% |
| dfdc_only_control | real.jpeg | REAL | FAKE | 84.87% / 15.13% |
| mixed_domains | random-person.jpeg | FAKE | FAKE | 68.74% / 31.26% |
| mixed_domains | real.jpeg | REAL | FAKE | 75.95% / 24.05% |

## Verified live preview

![Live Grad-CAM result and collapsed bottom notes](demo_live.png)
