"""Model pipeline, nested cross-validation, and final training."""

from collections import Counter

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PARAMETERS = {
    "randomforestclassifier__max_depth": [5, None],
    "randomforestclassifier__criterion": ["gini", "entropy"],
    "randomforestclassifier__min_samples_leaf": [1, 3],
}


def new_model():
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=0.99),
        RandomForestClassifier(n_estimators=50, max_features="sqrt", random_state=42),
    )


def train_and_evaluate(x, y):
    """Return unbiased predictions plus one fitted explanatory model."""
    outer = StratifiedKFold(10, shuffle=True, random_state=42)
    inner = StratifiedKFold(3, shuffle=True, random_state=42)
    predictions = np.empty_like(y)
    scores, choices = [], []

    for fold, (train, test) in enumerate(outer.split(x, y), 1):
        # One worker is reliable on Windows and is fast enough for this tiny dataset.
        search = GridSearchCV(new_model(), PARAMETERS, cv=inner, n_jobs=1)
        search.fit(x[train], y[train])
        predictions[test] = search.predict(x[test])
        score = np.mean(predictions[test] == y[test])
        scores.append(score)
        choices.append(search.best_params_)
        print(f"Fold {fold:2d}/10: {score:.3f}")

    common = Counter(tuple(sorted(p.items())) for p in choices).most_common(1)[0][0]
    parameters = dict(common)
    final_model = clone(new_model()).set_params(**parameters).fit(x, y)
    return {
        "predictions": predictions,
        "fold_scores": np.asarray(scores),
        "parameters": parameters,
        "model": final_model,
    }
