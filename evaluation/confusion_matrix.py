import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path,
):
    """
    Generates and saves a confusion matrix.

    Parameters
    ----------
    y_true : list
        Ground truth labels.

    y_pred : list
        Predicted labels.

    class_names : list
        Class names.

    save_path : Path
        Path where the figure is saved.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()