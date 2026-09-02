"""Run the Iris Random Forest experiment and publish its outputs."""

import os
from datetime import datetime, timezone
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from github_sync import GitError, sync_to_github
from model import train_and_evaluate
from plots import create_figures, show_figures


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
RESULTS = OUTPUT / "latest_results.txt"
PROJECT_FILES = [
    "main.py", "model.py", "plots.py", "github_sync.py",
    "README.md", "requirements.txt", ".gitignore",
]


def main():
    data = load_iris()
    result = train_and_evaluate(data.data, data.target)
    y_true, y_pred = data.target, result["predictions"]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=data.target_names)
    matrix = confusion_matrix(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}\n\n{report}\nConfusion matrix:\n{matrix}")

    OUTPUT.mkdir(exist_ok=True)
    RESULTS.write_text(
        f"Run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}\n"
        f"Accuracy: {accuracy:.4f}\n\n{report}\nConfusion matrix:\n{matrix}\n"
        f"\nBest parameters:\n{result['parameters']}\n",
        encoding="utf-8",
    )
    figures = create_figures(data, result, OUTPUT / "figures")
    print(f"\nSaved results and {len(figures)} figures in {OUTPUT}")

    if os.getenv("IRIS_SKIP_GITHUB_PUSH") != "1":
        files = [ROOT / name for name in PROJECT_FILES] + [RESULTS, *figures]
        try:
            sync_to_github(ROOT, files)
        except GitError as error:
            print(f"\nModel completed, but GitHub push failed:\n{error}")

    print("\nOpening all figures. Close the windows to finish.")
    show_figures()


if __name__ == "__main__":
    main()
