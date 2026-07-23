import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

from torch.amp import autocast
from models.vit import VisionTransformer
from datasets.dataloader import get_dataloaders

from evaluation.metrics import calculate_metrics
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.roc_curve import plot_roc_curve
from evaluation.precision_recall_curve import plot_precision_recall_curve

from utils.config import (
    DEVICE,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)


def evaluate():

    print("=" * 60)
    print("Evaluating Vision Transformer")
    print("=" * 60)

    # --------------------------------------------------
    # Load Test Data
    # --------------------------------------------------

    _, _, test_loader = get_dataloaders()

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    model = VisionTransformer().to(DEVICE)

    checkpoint = CHECKPOINT_DIR / "best_vit.pth"

    print("Checkpoint Used:")
    print(checkpoint.resolve())
    
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
        )
    )

    model.eval()

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with autocast(
                device_type=DEVICE.type,
                enabled=DEVICE.type =="cuda",
            ):
                
                outputs = model(images)

            probabilities = F.softmax(outputs, dim=1)

            predictions = outputs.argmax(dim=1)

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

            y_scores.extend(
                probabilities[:, 1].cpu().numpy()
            )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print("\nEvaluation Results\n")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # --------------------------------------------------
    # Save Metrics
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    with open(
        OUTPUT_DIR / "metrics.txt",
        "w",
    ) as file:

        for key, value in metrics.items():

            file.write(
                f"{key}: {value:.4f}\n"
            )

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Fake", "Real"],
    )

    print("\nClassification Report\n")
    print(report)

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
    ) as file:

        file.write(report)

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=["Fake", "Real"],
        save_path=OUTPUT_DIR / "confusion_matrix.png",
    )

    print("\nConfusion Matrix Saved")

    # --------------------------------------------------
    # ROC Curve
    # --------------------------------------------------

    plot_roc_curve(
        y_true,
        y_scores,
        OUTPUT_DIR / "roc_curve.png",
    )

    # --------------------------------------------------
    # Precision-Recall Curve
    # --------------------------------------------------

    plot_precision_recall_curve(
        y_true,
        y_scores,
        OUTPUT_DIR / "precision_recall_curve.png",
    )

    print("\nEvaluation Completed")


if __name__ == "__main__":
    evaluate()