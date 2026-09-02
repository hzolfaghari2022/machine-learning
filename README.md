# Modular Iris Random Forest

This compact project separates one job into each file:

- `main.py` — controls the experiment from beginning to end.
- `model.py` — builds the pipeline and runs nested cross-validation.
- `plots.py` — creates eight explanatory figures.
- `github_sync.py` — commits and pushes only this project's files.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The program saves everything under `output/`. After a successful run, it
automatically pushes the saved files to GitHub and then opens all eight figures
together. Close the figure windows to finish the program. Repository:
`https://github.com/hzolfaghari2022/machine-learning`.

Grid search uses one worker (`n_jobs=1`) for reliable repeated runs on Windows.
It tests eight parameter combinations with 3-fold inner validation, while the
final reported performance still uses the full 10-fold outer evaluation.

Git must be installed and authenticated. If Git needs your identity, configure
it once:

```bash
git config --global user.name "Hussein Zolfaghari"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

To deliberately test without pushing:

```bash
IRIS_SKIP_GITHUB_PUSH=1 python main.py
```
