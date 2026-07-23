import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)


def plot_precision_recall_curve(
    y_true,
    y_scores,
    save_path,
):
    """
    Plot Precision-Recall Curve.
    """

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_scores,
    )

    ap = average_precision_score(
        y_true,
        y_scores,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {ap:.4f}",
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

    print("Precision-Recall Curve Saved")
    print(f"Average Precision : {ap:.4f}")