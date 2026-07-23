import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def save_training_history(
    train_losses,
    train_accuracies,
    validation_accuracies,
    output_dir,
):
    """
    Saves training history as CSV and plots.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    # -----------------------------------------
    # Save CSV
    # -----------------------------------------

    history = pd.DataFrame(
        {
            "Epoch": epochs,
            "Train Loss": train_losses,
            "Train Accuracy": train_accuracies,
            "Validation Accuracy": validation_accuracies,
        }
    )

    history.to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    # -----------------------------------------
    # Loss Curve
    # -----------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        linewidth=2,
        label="Training Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(output_dir / "loss_curve.png")

    plt.close()

    # -----------------------------------------
    # Accuracy Curve
    # -----------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracies,
        marker="o",
        linewidth=2,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        validation_accuracies,
        marker="s",
        linewidth=2,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_dir / "accuracy_curve.png")

    plt.close()

    print("\nTraining history saved successfully!")