import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Column groups
NUMERIC_COLS = [
    "Age", "Year_of_Study", "Sleep_Hours", "Study_Hours_Per_Week", "GPA",
    "Social_Media_Hours", "Screen_Time_Hours", "Financial_Stress",
    "Family_Support", "Attendance_Rate",
]
ORDINAL_COLS = ["Physical_Activity", "Diet_Quality"]  # categories with a natural order
BINARY_COLS = ["Part_Time_Job", "Previous_Mental_Health_History"]  # No / Yes
# "Gender" is deliberately left out of the model (no signal, sensitive attribute)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Bounds are learned on the training data only (fit)."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def build_preprocessor():
    """Return the full preprocessing step (not yet fitted)."""

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper()),
        ("scaler", StandardScaler()),
    ])

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[
            ["Low", "Moderate", "High"],   # Physical_Activity
            ["Poor", "Average", "Good"],   # Diet_Quality
        ])),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("encoder", OrdinalEncoder(categories=[["No", "Yes"], ["No", "Yes"]])),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("ord", ordinal_pipe, ORDINAL_COLS),
            ("bin", binary_pipe, BINARY_COLS),
        ],
        remainder="drop",  # any other column (like Gender) is ignored
    )