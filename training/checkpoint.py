import torch
from pathlib import Path


def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_accuracy,
    train_losses,
    train_accuracies,
    validation_accuracies,
):
    """
    Save the complete training state.
    """

    checkpoint_path = Path(checkpoint_path)

    # Create directory if it doesn't exist
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_accuracy": best_accuracy,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "validation_accuracies": validation_accuracies,
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"\nCheckpoint saved at Epoch {epoch + 1}")


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
):
    """
    Load the complete training state.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():

        print("No checkpoint found.")

        return (
            0,
            0,
            [],
            [],
            [],
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_accuracy = checkpoint["best_accuracy"]

    train_losses = checkpoint["train_losses"]

    train_accuracies = checkpoint["train_accuracies"]

    validation_accuracies = checkpoint["validation_accuracies"]

    print("\nCheckpoint loaded successfully.")
    print(f"Resuming from Epoch {start_epoch + 1}")

    return (
        start_epoch,
        best_accuracy,
        train_losses,
        train_accuracies,
        validation_accuracies,
    )