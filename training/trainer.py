
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


def train(
    resume: bool = None,
    max_epochs: int = None,
):

    # ========================================================
    # DEVICE
    # ========================================================

    print(f"Using Device : {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA : {torch.version.cuda}"
        )

    epochs_to_run = (
        max_epochs
        if max_epochs is not None
        else EPOCHS
    )

    # ========================================================
    # DATA
    # ========================================================

    print("\nLoading Data...")

    train_loader, valid_loader, test_loader = (
        get_dataloaders()
    )

    print("Data Loaded Successfully.\n")

    # ========================================================
    # MODEL
    # ========================================================

    print("Creating Vision Transformer...")

    model = VisionTransformer().to(DEVICE)

    # channels_last can improve performance for models
    # containing convolution operations such as the ViT
    # patch embedding.
    if DEVICE.type == "cuda":

        model = model.to(
            memory_format=torch.channels_last
        )

    print("Model Created.\n")

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

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

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    USE_AMP = DEVICE.type == "cuda"

    scaler = GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    if USE_AMP:
        print("Mixed Precision : ENABLED")

    # ========================================================
    # HISTORY
    # ========================================================

    train_losses = []
    train_accuracies = []
    validation_accuracies = []

    best_accuracy = 0.0

    early_stopping = EarlyStopping(
        patience=5
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = (
        CHECKPOINT_DIR
        / "training_checkpoint.pth"
    )

    start_epoch = 0

    # ========================================================
    # RESUME
    # ========================================================

    if checkpoint_path.exists():

        if resume is None:

            try:

                choice = input(
                    "\nCheckpoint found. "
                    "Resume training? (y/n): "
                ).strip().lower()

                should_resume = (
                    choice == "y"
                )

            except EOFError:

                should_resume = True

        else:

            should_resume = resume

        if should_resume:

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

            print(
                f"\nResuming from Epoch "
                f"{start_epoch + 1}"
            )

        else:

            print(
                "\nStarting new training..."
            )

    # ========================================================
    # TRAINING
    # ========================================================

    print("=" * 60)
    print("Training Started")
    print("=" * 60)

    for epoch in range(
        start_epoch,
        epochs_to_run,
    ):

        model.train()

        # ----------------------------------------------------
        # Accumulators stay on GPU.
        #
        # This avoids calling .item() every batch.
        # ----------------------------------------------------

        running_loss = torch.zeros(
            1,
            device=DEVICE,
        )

        correct = torch.zeros(
            1,
            device=DEVICE,
        )

        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{epochs_to_run}"
            ),
            leave=True,
        )

        # ----------------------------------------------------
        # TRAIN BATCHES
        # ----------------------------------------------------

        for step, (images, labels) in enumerate(
            progress_bar
        ):

            # ------------------------------------------------
            # Move data efficiently to GPU
            # ------------------------------------------------

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            # channels_last for image tensors
            if DEVICE.type == "cuda":

                images = images.contiguous(
                    memory_format=torch.channels_last
                )

            # ------------------------------------------------
            # Gradient reset
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            with autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            scaler.scale(loss).backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            scaler.step(optimizer)

            scaler.update()

            # ------------------------------------------------
            # Statistics
            #
            # Keep these on GPU to avoid synchronization.
            # ------------------------------------------------

            batch_size = labels.size(0)

            running_loss += (
                loss.detach() * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum()

            total += batch_size

            # ------------------------------------------------
            # Update tqdm only every 25 batches.
            #
            # Updating tqdm every single batch creates
            # unnecessary CPU overhead.
            # ------------------------------------------------

            if (
                step % 25 == 0
                or step == len(train_loader) - 1
            ):

                progress_bar.set_postfix(
                    loss=f"{loss.detach().item():.4f}"
                )

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        train_loss = (
            running_loss.item()
            / total
        )

        train_accuracy = (
            100.0
            * correct.item()
            / total
        )

        train_losses.append(
            train_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        validation_correct = torch.zeros(
            1,
            device=DEVICE,
        )

        validation_total = 0

        with torch.inference_mode():

            for images, labels in tqdm(
                valid_loader,
                desc="Validation",
                leave=False,
            ):

                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True,
                )

                if DEVICE.type == "cuda":

                    images = images.contiguous(
                        memory_format=torch.channels_last
                    )

                with autocast(
                    device_type=DEVICE.type,
                    enabled=USE_AMP,
                ):

                    outputs = model(images)

                predictions = outputs.argmax(
                    dim=1
                )

                validation_correct += (
                    predictions == labels
                ).sum()

                validation_total += (
                    labels.size(0)
                )

        validation_accuracy = (
            100.0
            * validation_correct.item()
            / validation_total
        )

        validation_accuracies.append(
            validation_accuracy
        )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(
            validation_accuracy
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ====================================================
        # RESULTS
        # ====================================================

        print("\n" + "-" * 60)

        print(
            f"Epoch : "
            f"{epoch + 1}/{epochs_to_run}"
        )

        print(
            f"Training Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Training Accuracy   : "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Validation Accuracy : "
            f"{validation_accuracy:.2f}%"
        )

        print(
            f"Learning Rate       : "
            f"{current_lr:.8f}"
        )

        print("-" * 60)

        # ====================================================
        # BEST MODEL
        # ====================================================

        if validation_accuracy > best_accuracy:

            best_accuracy = (
                validation_accuracy
            )

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR
                / "best_vit.pth",
            )

            print(
                "Best Model Saved."
            )

        # ====================================================
        # CHECKPOINT
        # ====================================================

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

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        early_stopping(
            validation_accuracy
        )

        if early_stopping.early_stop:

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # COMPLETE
    # ========================================================

    print("=" * 60)

    print("Training Completed")

    print(
        f"Best Validation Accuracy : "
        f"{best_accuracy:.2f}%"
    )

    print("=" * 60)

    save_training_history(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        validation_accuracies=validation_accuracies,
        output_dir="outputs",
    )

    return model
