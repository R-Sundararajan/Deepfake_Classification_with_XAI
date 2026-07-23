import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
)


def plot_roc_curve(
    y_true,
    y_scores,
    save_path,
):
    """
    Plot ROC Curve and save it.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_scores,
    )

    auc_score = roc_auc_score(
        y_true,
        y_scores,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc_score:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("Receiver Operating Characteristic (ROC)")

    plt.legend(loc="lower right")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

    print(f"ROC Curve Saved")
    print(f"AUC Score : {auc_score:.4f}")
    