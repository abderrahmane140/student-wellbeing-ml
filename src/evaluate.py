import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

import json
from pathlib import Path

LABELS = ["Low", "Medium", "High"]

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "roc_auc": "roc_auc_ovr",
}


def get_cv():
    """5 folds that keep the class proportions. Fixed seed = reproducible."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def compare_models(models, X, y):
    """Cross-validate every model and return one table of average scores."""
    rows = []
    for name, pipe in models.items():
        res = cross_validate(
            pipe, X, y, cv=get_cv(), scoring=SCORING, return_train_score=True
        )
        rows.append({
            "model": name,
            "accuracy": res["test_accuracy"].mean(),
            "precision": res["test_precision"].mean(),
            "recall": res["test_recall"].mean(),
            "f1": res["test_f1"].mean(),
            "roc_auc": res["test_roc_auc"].mean(),
            "train_accuracy": res["train_accuracy"].mean(),
        })
    return pd.DataFrame(rows).set_index("model").round(3)


def cv_confusion_matrix(pipe, X, y):
    """Confusion matrix built from cross-validated predictions."""
    y_pred = cross_val_predict(pipe, X, y, cv=get_cv())
    return confusion_matrix(y, y_pred, labels=LABELS) 


def evaluate_on_test(pipe, X_test, y_test):
    """Score a fitted pipeline on the test set. Returns (metrics, predictions)."""
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr", labels=pipe.classes_),
    }
    return {k: round(v, 3) for k, v in metrics.items()}, y_pred


def save_metrics(metrics, cm, path):
    """Save test metrics and the confusion matrix as a JSON file."""
    path  = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "metrics": metrics,
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
    }
    with open(path, "w") as f:
        json.dump(content, f, indent=2)
