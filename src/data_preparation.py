from pathlib import Path

import pandas as pd 
from sklearn.model_selection import train_test_split


# Project root = the folder above src/
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "student_mental_health.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

TARGET = "Mental_Health_Risk"
RANDOM_STATE = 42

def load_data(path=DATA_PATH):
    """load the raw CSV file."""
    if not Path(path).exists():
        raise FileExistsError(f"Data file not found: {path}")
    return pd.read_csv(path)


def split_data(df):
    """Split into 80% train / 20% test, keeping class proportions."""
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' is missing from the data")

    # Student_ID is only a label, so it's not a feature
    X = df.drop(columns=[TARGET, "Student_ID"])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )   
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, folder=PROCESSED_DIR):
    """Save the 4 splits as CSV files in data/processed/."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(folder / "X_train.csv", index=False)
    X_test.to_csv(folder / "X_test.csv", index=False)
    y_train.to_csv(folder / "y_train.csv", index=False)
    y_test.to_csv(folder / "y_test.csv", index=False)


def load_splits(folder=PROCESSED_DIR):
    """Load the 4 splits saved by save_splits()."""
    folder = Path(folder)
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        if not (folder / f"{name}.csv").exists():
            raise FileNotFoundError(f"{name}.csv not found in {folder}. Run save_splits first.")

    X_train = pd.read_csv(folder / "X_train.csv")
    X_test = pd.read_csv(folder / "X_test.csv")
    y_train = pd.read_csv(folder / "y_train.csv")[TARGET]
    y_test = pd.read_csv(folder / "y_test.csv")[TARGET]
    return X_train, X_test, y_train, y_test
