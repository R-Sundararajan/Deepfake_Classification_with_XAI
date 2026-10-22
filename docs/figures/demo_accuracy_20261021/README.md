# Image accuracy plot evidence — 21 October 2026

These aggregate PNG/SVG plots and `metrics.json` are portable documentation assets. No face images, checkpoints or per-image paths are included. The original experiments and their plots remain unchanged.

## Accuracy and error comparison

Why: quantify the improvement on equal evaluation splits. The left panel shows accuracy on a zero-based scale; the right panel shows the corresponding number of errors. Each split contains 20,000 images.

[PNG](accuracy_comparison.png) · [SVG](accuracy_comparison.svg)

## Training and validation curves

Why: show how continued training changed accuracy and loss. The dashed line separates the original two epochs from four continuation epochs; the star marks the validation-selected epoch. Training accuracy includes stochastic augmentation. No test scores selected the epoch.

[PNG](training_curves.png) · [SVG](training_curves.svg)

## Confusion matrices

Why: show which class errors remain. Rows are actual labels and columns are predictions; off-diagonal cells are mistakes. Counts and within-class percentages use a shared color scale. The original test matrix is previously recorded evidence.

[PNG](confusion_matrices.png) · [SVG](confusion_matrices.svg)

## ROC and precision–recall

Why: demonstrate separation and the precision/recall trade-off across thresholds. FAKE is positive; dots mark the deployed threshold of 0.5. Dashed references show chance ROC and the 50% FAKE prevalence. Insets magnify the near-perfect region with explicitly zoomed axes. These are descriptive plots from saved scores; no threshold was selected on test data.

[PNG](roc_precision_recall.png) · [SVG](roc_precision_recall.svg)

## Reproduce

From the project root, using existing local experiment JSON/CSV logs:

```powershell
.\dfenv\Scripts\python.exe -B -m tools.plot_demo_accuracy
```

To render the same figures from the compact aggregate snapshot without local experiment files:

```powershell
.\dfenv\Scripts\python.exe -B -m tools.plot_demo_accuracy --snapshot docs/figures/demo_accuracy_20261021/metrics.json
```

This reads saved scores only and does not run inference, retrain, or re-evaluate test images. Each saved score CSV is checked against its recorded sample count, confusion matrix, accuracy and ROC-AUC. Curve rendering uses at most 1,200 sampled points; reported metrics come from all 20,000 scores. Source artifact SHA-256 hashes and the selected checkpoint SHA-256 are recorded in `metrics.json`.
