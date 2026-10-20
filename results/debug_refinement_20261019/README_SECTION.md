## Debugging and demo refinement — 19 October 2026

Full evidence, all 38 supplied-image runs (19 configurations), saved crops/tensors, exact commands and remaining limitations: [executed debugging report](results/debug_refinement_20261019/REPORT.md).

Verified: S2 reproduces the synthetic-image failure (16.13% FAKE); its 81.40% video balanced accuracy and 0.518 macro-F1 are consistent with `[[11,0],[61,103]]` on the imbalanced validation set. Checkpoint/class mapping is correct. S5 reproduces the reported 66.15% REAL example. Domain transfer and native preprocessing matter: applying DFDC alignment to the image-trained ResNet reverses its synthetic sanity prediction. A new separate S2 image validation benchmark measures 51.10% balanced accuracy; its video score is not an image score.

Fixed: legacy CLI bicubic/bilinear mismatch, IG sign loss and insufficient integration diagnostics, blue tint on zero maps, Grad-CAM++ small-gradient cutoff, and repeated interaction work. Existing deployed image scores were preserved exactly in all before/after pairs. Threshold tuning was audited on validation only and was not stable enough to change deployment. Historical test outputs remain untouched.

### Image-family ranking (common 2,000-image validation cohort)

| Family | Best variant | Best validation BA | Mean validation BA | Comparable runs |
|---|---|---|---|---|
| ResNet / CNN | S0 | 96.90% | 93.85% | 2 |
| EfficientNet | E02_efficientnet_b0 | 92.45% | 81.96% | 4 |

### Video-family ranking (common grouped DFDC validation cohort)

| Family | Best variant | Best validation BA | Mean validation BA | Comparable runs |
|---|---|---|---|---|
| Spatial + Temporal Hybrid | S6 | 94.82% | 94.61% | 3 |
| EfficientNet | S1b | 81.40% | 74.07% | 4 |
| Vision Transformer | S4 | 78.69% | 70.51% | 2 |
| CNN + ViT Hybrid | S5 | 72.31% | 72.31% | 1 |
| ResNet / CNN | S0 | 50.00% | 50.00% | 1 |

Families rank by their best validated compatible variant. Means count only completed comparable runs. S0 trained on 100,000 images, E01–E04 on 12,000, and S2 on DFDC videos; training budgets/domains differ. S0 remains the image default and S6 the video default. Unbenchmarked image transfers are optional and unranked. Model details expose datasets, checkpoint hashes, accuracy, macro-F1 and limitations.

## Explanation and uncertainty interface

CNNs: Grad-CAM, Grad-CAM++ (first-gradient-power approximation), Layer-CAM, signed IG, Occlusion Sensitivity and on-demand LIME. ViTs: IG, Occlusion, LIME, native attention rollout and gradient-weighted attention. Hybrids offer their compatible methods. Every explanation has a one-line description, REAL/FAKE target, color legend and opacity control. Attention is not causal proof; hybrid CAM localizes the CNN branch. Video perturbation/gradient XAI differentiates the selected complete decision with its video temperature and other frame embeddings held fixed.

IG uses batched Gauss–Legendre integration and exposes mean-color or blurred-input reference, signed raw statistics and completeness/convergence. Across 368 actual image/method/target checks on 15 deployed checkpoints, all maps were finite and scores unchanged. At 128 steps, 18 IG cases were numerically inconclusive; 2 maps were constant and 18 LIME fits weak. The UI flags them. Negative deletion results remain in the logs; these checks do not establish universal faithfulness.

Predictions and completed explanations use bounded Streamlit caches keyed by input/checkpoint, modality, method, target and parameters. Uploading classifies without XAI; Generate explanation computes only the selected method. Opacity changes reuse the map. Typical resident S2 synthetic prediction with Grad-CAM improved from 146.95 to 126.79 ms; prediction alone is 87.02 ms, and a cached prediction lookup is 0.89 ms. Timings exclude browser paint; cold GPU startup remains slower. [Detailed profiling](results/debug_refinement_20261019/profile.json).

REAL/FAKE MP4 video QA covered every frozen system (22 runs), deterministic eight-frame sampling, complete-system XAI and calibration parity. Lossless MOV remux, playback/preview controls and corrupt/missing input handling were checked. S6 correctly labels both selected clips; S0 still misses the FAKE clip. Notes and warnings, including temporal image fallbacks, remain in a collapsed section at the bottom.

The registered mixed-domain pilot used 2,000 training images and existing DFDC training data with balanced, equal domain losses and a frozen encoder:

| Run | Image validation BA | Image macro-F1 | Video validation BA | Video macro-F1 |
|---|---|---|---|---|
| baseline_S2 | 51.10% | 0.511 | 81.40% | 0.518 |
| dfdc_only_control | 53.00% | 0.528 | 79.57% | 0.495 |
| mixed_domains | 73.35% | 0.733 | 82.32% | 0.530 |

It improves S2 transfer but fails the registered joint champion rule, so it remains a research artifact. No supplied sanity image or test data entered training/selection. [Pilot registration and results](results/debug_refinement_20261019/domain_pilot/comparison.json).

Regression: 56 tests passed. Actual UI/upload/cache checks: 19 passed. Preserved originals use SHA-256-addressed local upload paths; metadata establishes provenance, not manipulation. Small source cohorts, crop/domain shift, weak/inconclusive attributions, shared validation calibration and cold startup remain limitations.
