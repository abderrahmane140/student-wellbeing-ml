from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path

import joblib
from src.preprocessing import build_preprocessor

from src.data_preparation import load_data, split_data, save_splits
from src.evaluate import evaluate_on_test

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.evaluate import get_cv



RANDOM_STATE = 42


def build_pipeline(model):
    """Chain the preprocessing step and a model into one pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def get_models():
    """Return the models to compare, each inside its own pipeline."""
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
    }


# ---------- Hyperparameter search ----------


LR_PARAMS = {
    "model__C": [0.01, 0.1, 1, 10, 100],
}

GB_PARAMS = {
    "model__n_estimators": [50, 100, 200],     # Number of boosting trees
    "model__learning_rate": [0.01, 0.05, 0.1], # Contribution of each tree
    "model__max_depth": [1, 2, 3],             # Maximum depth of each tree
    "model__subsample": [0.7, 1.0],            # Fraction of samples used for each boosting iteration
    "model__min_samples_leaf": [1, 10, 30],    # Minimum samples required in a leaf
}


def tune_logistic_regression(X, y):
    """GridSearchCV: tests every C value."""
    search = GridSearchCV(
        get_models()["Logistic Regression"],
        param_grid=LR_PARAMS,
        cv=get_cv(),
        scoring="f1_macro",
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def tune_gradient_boosting(X, y):
    """RandomizedSearchCV: tests 20 random combinations."""
    search = RandomizedSearchCV(
        get_models()["Gradient Boosting"],
        param_distributions=GB_PARAMS,
        n_iter=20,
        cv=get_cv(),
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X, y)
    return search


# ---------- Final model ----------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "student_risk_pipeline.joblib"


def train_final_model(X_train, y_train):
    pipe = build_pipeline(
        LogisticRegression(C=10, max_iter=1000, random_state=RANDOM_STATE)
    )
    pipe.fit(X_train, y_train)
    return pipe


def save_model(pipe, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def load_model(path=MODEL_PATH):
    """Load the saved pipeline."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}. Run 'python -m src.train' first.")
    return joblib.load(path)


def main():
    """Run the whole pipeline: load, split, train, evaluate, save."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test)

    pipe = train_final_model(X_train, y_train)
    metrics, _ = evaluate_on_test(pipe, X_test, y_test)
    print("Test metrics:", metrics)

    path = save_model(pipe)
    print("Model saved to:", path)


if __name__ == "__main__":
    main()