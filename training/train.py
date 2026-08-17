"""Train the ResNet-18 baseline with checkpointing and early stopping."""

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_OUTPUT_PATH,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
    TRAINABLE_BACKBONE_BLOCKS,
    WEIGHT_DECAY,
)
from models.classifier import create_model, parameter_counts
from training.data import build_dataloaders, inspect_dataset, print_dataset_report


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        progress = tqdm(loader, desc="train" if training else "valid", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_count = labels.size(0)
            loss_sum += loss.detach().item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_count
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    return loss_sum / total, correct / total


def _save_history(history: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["valid_accuracy"] for row in history], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)


def train(args: argparse.Namespace) -> Path:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    report = inspect_dataset(save_report=True)
    print_dataset_report(report)
    train_loader, valid_loader, _, class_to_idx = build_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
    )

    model = create_model(pretrained=True).to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    print(f"Model: {MODEL_NAME} (ImageNet pretrained)")
    print(f"Parameters: {total_parameters:,} total; {trainable_parameters:,} trainable")
    print(f"Fine-tuned residual blocks: {TRAINABLE_BACKBONE_BLOCKS}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []
    start_time = time.perf_counter()
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        valid_loss, valid_accuracy = _run_epoch(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}: train loss {train_loss:.4f}, "
            f"train accuracy {train_accuracy:.4%}, valid loss {valid_loss:.4f}, "
            f"valid accuracy {valid_accuracy:.4%}, time {epoch_seconds:.1f}s"
        )

        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "image_size": IMAGE_SIZE,
                    "class_to_idx": class_to_idx,
                    "trainable_backbone_blocks": TRAINABLE_BACKBONE_BLOCKS,
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                    "training_seconds_so_far": time.perf_counter() - start_time,
                    "seed": SEED,
                },
                MODEL_OUTPUT_PATH,
            )
            print(f"Saved best checkpoint: {MODEL_OUTPUT_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping: validation loss did not improve.")
                break

    _save_history(history)
    print(f"Training completed in {time.perf_counter() - start_time:.1f}s")

    if not args.skip_test_evaluation:
        from evaluation.evaluate import evaluate_checkpoint

        evaluate_checkpoint(
            checkpoint_path=MODEL_OUTPUT_PATH,
            split="test",
            batch_size=args.batch_size,
            max_samples=args.max_test_samples,
        )
    return MODEL_OUTPUT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
