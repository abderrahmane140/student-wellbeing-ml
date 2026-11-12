import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.data_preparation import load_data, split_data, save_splits, load_splits
from src.evaluate import LABELS, evaluate_on_test, save_metrics
from src.preprocessing import build_preprocessor
from src.train import load_model, save_model, train_final_model

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"

def prepare_data():
    """Task 1: load the raw data, split it, save the splits."""
    df = load_data()
    if df.empty:
        raise ValueError("The raw data file is empty")

    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test)
    print(f"Splits saved: train={len(X_train)} rows, test={len(X_test)} rows")



def check_preprocessing():
    """Task 2: fit the preprocessor on train and check the output is clean."""
    X_train, _, _, _ = load_splits()

    X_prep = build_preprocessor().fit_transform(X_train)
    if np.isnan(X_prep).any():
        raise ValueError("Missing values remain after preprocessing")

    print("Preprocessing OK, output shape:", X_prep.shape)


def train_model():
    """Task 3: train the final pipeline and save it."""
    X_train, _, y_train, _ = load_splits()

    pipe = train_final_model(X_train, y_train)
    path = save_model(pipe)
    print("Model saved to:", path)


def evaluate_model():
    """Task 4: score the saved model on the test set and save the results."""
    _, X_test, _, y_test = load_splits()
    pipe = load_model()

    metrics, y_pred = evaluate_on_test(pipe, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    save_metrics(metrics, cm, METRICS_PATH)

    print("Test metrics:", metrics)
    print("Results saved to:", METRICS_PATH)


STEPS = {
    "prepare": prepare_data,
    "preprocess": check_preprocessing,
    "train": train_model,
    "evaluate": evaluate_model,
}

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else None

    if choice == "all":
        for step in STEPS.values():
            step()
    elif choice in STEPS:
        STEPS[choice]()
    else:
        print("Usage: python -m src.pipeline_steps [prepare|preprocess|train|evaluate|all]")
        sys.exit(1)