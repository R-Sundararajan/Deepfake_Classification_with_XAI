# Current image figures

All metrics are derived from saved predictions. Training stages and comparison scopes are explicit.

- [resnet18_accuracy PNG](resnet18_accuracy.png) · [editable SVG](resnet18_accuracy.svg): Training and full-validation accuracy across trustworthy global epochs; selected epoch marked.
- [resnet18_loss PNG](resnet18_loss.png) · [editable SVG](resnet18_loss.svg): Observed global-epoch losses; stage boundaries and early stopping shown where applicable.
- [efficientnet_b0_accuracy PNG](efficientnet_b0_accuracy.png) · [editable SVG](efficientnet_b0_accuracy.svg): Only the new 100k-image stage; the earlier 12k-image warm start is excluded from this axis.
- [efficientnet_b0_loss PNG](efficientnet_b0_loss.png) · [editable SVG](efficientnet_b0_loss.svg): Training and validation loss over the full-dataset stage.
- [validation_comparison PNG](validation_comparison.png) · [editable SVG](validation_comparison.svg): Two frozen models on the same full 20,000-image validation split.
- [test_comparison PNG](test_comparison.png) · [editable SVG](test_comparison.svg): Repeated historically used test split after freezing both selections; no test-based tuning.
- [confusion_matrices PNG](confusion_matrices.png) · [editable SVG](confusion_matrices.svg): REAL then FAKE row/column order; off-diagonals are errors at threshold 0.5.
- [roc_curves PNG](roc_curves.png) · [editable SVG](roc_curves.svg): FAKE-positive ROC from every saved validation/test probability.
- [precision_recall_curves PNG](precision_recall_curves.png) · [editable SVG](precision_recall_curves.svg): FAKE-positive precision–recall; legends report average precision.
- [classwise_metrics PNG](classwise_metrics.png) · [editable SVG](classwise_metrics.svg): Precision, recall and F1 for each class and split.
- [model_resources PNG](model_resources.png) · [editable SVG](model_resources.svg): Actual parameter counts and selected checkpoint bytes, including metadata.
- [inference_comparison PNG](inference_comparison.png) · [editable SVG](inference_comparison.svg): Identical FP32 resident-input timing protocol with warmup and synchronization.
- [validation_improvement PNG](validation_improvement.png) · [editable SVG](validation_improvement.svg): Both existing and updated checkpoints evaluated on the same full validation split.
- [resnet18_test_improvement PNG](resnet18_test_improvement.png) · [editable SVG](resnet18_test_improvement.svg): Recorded epoch-6 versus current frozen ResNet results on the same historically used 20,000 test images; descriptive only, never test-based selection.

Regenerate using `python -B -m tools.plot_demo_accuracy --current`. A portable regeneration uses `--snapshot docs/figures/current_image_models/metrics.json`. No inference or training runs during plotting.
