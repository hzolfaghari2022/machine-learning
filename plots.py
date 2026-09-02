"""Small collection of figures that explain the data, method, and results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree


COLORS = ["#0072B2", "#E69F00", "#009E73"]


def save(fig, folder, name):
    path = folder / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return path


def show_figures():
    """Open all saved figures together after the experiment is complete."""
    plt.show()


def create_figures(data, result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    x, y, names = data.data, data.target, data.target_names
    pred, scores = result["predictions"], result["fold_scores"]
    files = []

    # 1. Program workflow.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    steps = ["Load Iris", "10-fold test split", "Scale → PCA → Forest",
             "Inner grid search", "Predict and evaluate", "Save and push"]
    for i, text in enumerate(steps):
        y_pos = .9 - i * .16
        ax.text(.5, y_pos, text, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=.6", fc="#EAF2F8", ec="#2874A6"))
        if i < len(steps) - 1:
            ax.annotate("", (.5, y_pos - .11), (.5, y_pos - .05),
                        arrowprops=dict(arrowstyle="->", color="#2874A6", lw=2))
    ax.set_title("How the Program Works", fontsize=17, weight="bold")
    files.append(save(fig, folder, "01_workflow.png"))

    # 2. One dashboard for class balance and all four input features.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].bar(names, np.bincount(y), color=COLORS)
    axes[0, 0].set_title("Class balance")
    for j, ax in enumerate(axes.flat[1:5]):
        for c, name in enumerate(names):
            ax.hist(x[y == c, j], bins=9, alpha=.5, color=COLORS[c], label=name)
        ax.set_title(data.feature_names[j])
    for c, name in enumerate(names):
        axes[1, 2].scatter(x[y == c, 2], x[y == c, 3], color=COLORS[c], label=name)
    axes[1, 2].set(title="Petal separation", xlabel="Petal length", ylabel="Petal width")
    axes[1, 2].legend()
    fig.suptitle("Understanding the Iris Data", fontsize=16, weight="bold")
    fig.tight_layout()
    files.append(save(fig, folder, "02_data_overview.png"))

    # Shared PCA coordinates for the next two figures.
    scaled = StandardScaler().fit_transform(x)
    pca = PCA().fit(scaled)
    points = pca.transform(scaled)

    # 3. PCA variance and two-dimensional view.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    variance = pca.explained_variance_ratio_ * 100
    axes[0].bar(range(1, 5), variance, color="#56B4E9")
    axes[0].plot(range(1, 5), np.cumsum(variance), "o-", color="#D55E00")
    axes[0].set(title="PCA explained variance", xlabel="Component", ylabel="Percent")
    for c, name in enumerate(names):
        axes[1].scatter(points[y == c, 0], points[y == c, 1],
                        color=COLORS[c], label=name, alpha=.75)
    axes[1].set(title="First two PCA components", xlabel="PC1", ylabel="PC2")
    axes[1].legend()
    fig.tight_layout()
    files.append(save(fig, folder, "03_pca.png"))

    # 4. Test accuracy in each outer fold.
    fig, ax = plt.subplots(figsize=(9, 5))
    folds = np.arange(1, 11)
    ax.plot(folds, scores * 100, "o-", lw=2)
    ax.axhline(scores.mean() * 100, color="#D55E00", ls="--", label="Mean")
    ax.set(title="Outer-fold test accuracy", xlabel="Fold", ylabel="Accuracy (%)",
           xticks=folds, ylim=(0, 105))
    ax.legend()
    files.append(save(fig, folder, "04_fold_accuracy.png"))

    # 5. Confusion matrix.
    matrix = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True", title="Out-of-fold confusion matrix")
    for row, col in np.ndindex(matrix.shape):
        ax.text(col, row, matrix[row, col], ha="center", va="center", fontsize=13)
    files.append(save(fig, folder, "05_confusion_matrix.png"))

    # 6. Precision, recall, and F1-score.
    metrics = precision_recall_fscore_support(y, pred, zero_division=0)[:3]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions, width = np.arange(3), .25
    for i, (values, label) in enumerate(zip(metrics, ["Precision", "Recall", "F1"])):
        ax.bar(positions + (i - 1) * width, values, width, label=label)
    ax.set(xticks=positions, xticklabels=names, ylim=(0, 1.08), title="Metrics by class")
    ax.legend(ncol=3)
    files.append(save(fig, folder, "06_class_metrics.png"))

    # 7. Show where out-of-fold errors occur.
    correct = pred == y
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(points[correct, 0], points[correct, 1], alpha=.45, label="Correct")
    ax.scatter(points[~correct, 0], points[~correct, 1], marker="X", s=90,
               color="#D55E00", label="Incorrect")
    ax.set(title="Prediction errors in PCA space", xlabel="PC1", ylabel="PC2")
    ax.legend()
    files.append(save(fig, folder, "07_prediction_errors.png"))

    # 8. One tree from the fitted forest.
    model = result["model"]
    forest, fitted_pca = model[-1], model[-2]
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(forest.estimators_[0], max_depth=3,
              feature_names=[f"PC{i + 1}" for i in range(fitted_pca.n_components_)],
              class_names=list(names), filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("One Tree from the Random Forest (first 3 levels)")
    files.append(save(fig, folder, "08_example_tree.png"))
    return files
