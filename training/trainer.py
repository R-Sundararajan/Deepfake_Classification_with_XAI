import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.amp import autocast, GradScaler
from models.vit import VisionTransformer
from datasets.dataloader import get_dataloaders

from utils.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
)

from training.plot_history import save_training_history
from training.early_stopping import EarlyStopping
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def train():

    # ----------------------------------------------------
    # Device
    # ----------------------------------------------------

    print(f"Using Device : {DEVICE}")

    # ----------------------------------------------------
    # Data
    # ----------------------------------------------------

    print("\nLoading Data...")

    train_loader, valid_loader, test_loader = get_dataloaders()

    print("Data Loaded Successfully.\n")

    # ----------------------------------------------------
    # Model
    # ----------------------------------------------------

    print("Creating Vision Transformer...")

    model = VisionTransformer().to(DEVICE)

    print("Model Created.\n")

    # ----------------------------------------------------
    # Loss Function
    # ----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------
    # Optimizer
    # ----------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    # ----------------------------------------------------
    # Mixed Precision
    # ----------------------------------------------------

    USE_AMP = DEVICE.type == "cuda"

    scaler = GradScaler(
        device=DEVICE.type,
        enabled=USE_AMP,
    )
    # ----------------------------------------------------
    # Training History
    # ----------------------------------------------------

    train_losses = []
    train_accuracies = []
    validation_accuracies = []

    best_accuracy = 0

    early_stopping = EarlyStopping(
        patience=5
    )

    print("=" * 60)
    print("Training Started")
    print("=" * 60)

    # ----------------------------------------------------
    # Resume Training
    # ----------------------------------------------------

    checkpoint_path = CHECKPOINT_DIR / "training_checkpoint.pth"

    start_epoch = 0

    if checkpoint_path.exists():

        choice = input(
            "\nCheckpoint found. Resume training? (y/n): "
        ).strip().lower()

        if choice == "y":

            (
                start_epoch,
                best_accuracy,
                train_losses,
                train_accuracies,
                validation_accuracies,
            ) = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
            )

            print(f"\nResuming from Epoch {start_epoch + 1}")

        else:

            print("\nStarting new training...")

    # ====================================================
    # Training Loop
    # ====================================================

    for epoch in range(start_epoch, EPOCHS):

        model.train()

        running_loss = 0

        correct = 0

        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=True,
        )

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
)

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / len(train_loader)

        train_accuracy = 100 * correct / total

        train_losses.append(train_loss)

        train_accuracies.append(train_accuracy)

        # ------------------------------------------------
        # Validation
        # ------------------------------------------------

        model.eval()

        correct = 0

        total = 0

        with torch.no_grad():

            for images, labels in tqdm(
                valid_loader,
                desc="Validation",
                leave=False,
            ):

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                with autocast(
                    device_type=DEVICE.type,
                    enabled=USE_AMP,
                ):

                    outputs = model(images)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

        validation_accuracy = 100 * correct / total

        validation_accuracies.append(validation_accuracy)

        scheduler.step(validation_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "-" * 60)

        print(f"Epoch : {epoch + 1}/{EPOCHS}")

        print(f"Training Loss        : {train_loss:.4f}")

        print(f"Training Accuracy    : {train_accuracy:.2f}%")

        print(f"Validation Accuracy  : {validation_accuracy:.2f}%")

        print(f"Learning Rate        : {current_lr:.8f}")

        print("-" * 60)

        # ------------------------------------------------
        # Save Best Model
        # ------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_vit.pth",
            )

            print("Best Model Saved.")

        # ------------------------------------------------
        # Save Training Checkpoint
        # ------------------------------------------------

        save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_accuracy=best_accuracy,
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            validation_accuracies=validation_accuracies,
        )

        # ------------------------------------------------
        # Early Stopping
        # ------------------------------------------------

        early_stopping(validation_accuracy)

        if early_stopping.early_stop:

            print("\nEarly stopping triggered.")

            break

    print("=" * 60)

    print("Training Completed")

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

    print("=" * 60)

    save_training_history(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        validation_accuracies=validation_accuracies,
        output_dir="outputs",
    )

    return model