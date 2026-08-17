"""Evaluate a trained checkpoint and save concise classification metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from config import BATCH_SIZE, MODEL_OUTPUT_PATH, OUTPUT_DIR
from models.classifier import load_model_checkpoint
from training.data import build_dataloaders


def evaluate_checkpoint(
    checkpoint_path: str | Path = MODEL_OUTPUT_PATH,
    split: str = "test",
    batch_size: int = BATCH_SIZE,
    max_samples: int | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    train_loader, valid_loader, test_loader, class_to_idx = build_dataloaders(
        batch_size=batch_size,
        max_train_samples=max_samples if split == "train" else 2,
        max_valid_samples=max_samples if split == "valid" else 2,
        max_test_samples=max_samples if split == "test" else 2,
    )
    loader = {"train": train_loader, "valid": valid_loader, "test": test_loader}[split]

    y_true: list[int] = []
    y_pred: list[int] = []
    fake_scores: list[float] = []
    fake_index = class_to_idx["fake"]
    real_index = class_to_idx["real"]

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"evaluate {split}", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            fake_scores.extend(probabilities[:, fake_index].cpu().numpy().tolist())

    labels_order = [real_index, fake_index]
    matrix = confusion_matrix(y_true, y_pred, labels=labels_order)
    metrics = {
        "split": split,
        "samples": len(y_true),
        "positive_class": "FAKE",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=fake_index, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=fake_index, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=fake_index, zero_division=0),
        "roc_auc": roc_auc_score(np.asarray(y_true) == fake_index, fake_scores),
        "confusion_matrix_labels": ["REAL", "FAKE"],
        "confusion_matrix": matrix.tolist(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_valid_accuracy": checkpoint.get("valid_accuracy"),
        "training_seconds": checkpoint.get("training_seconds_so_far"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=["REAL", "FAKE"],
        digits=4,
        zero_division=0,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    display = ConfusionMatrixDisplay(matrix, display_labels=["REAL", "FAKE"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({split.title()} Split)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics and confusion matrix to {OUTPUT_DIR}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_OUTPUT_PATH)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate_checkpoint(
        arguments.checkpoint, arguments.split, arguments.batch_size, arguments.max_samples
    )
