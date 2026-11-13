import json
import sys
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Make the project root importable (the saved model needs src.preprocessing)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


LABELS = ["Low", "Medium", "High"]
MODEL_PATH = ROOT_DIR / "models" / "student_risk_pipeline.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"

st.set_page_config(page_title="Student Wellbeing", page_icon="🧠")
st.title("🧠 Student Mental Health Risk")
st.warning(
    "Educational prototype only. This model is NOT a diagnostic tool, a clinical "
    "screening tool, or a basis for disciplinary or selection decisions."
)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run the Airflow DAG or 'python -m src.pipeline_steps train' first."
        )
    return joblib.load(MODEL_PATH)

def get_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "reports/metrics.json not found. Run the Airflow DAG or "
            "'python -m src.pipeline_steps evaluate' first."
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


# Load everything once, stop with a clear message if something is missing
try:
    model = get_model()
    results = get_metrics()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

tab_perf, tab_predict = st.tabs(["📊 Model performance", "🔮 Test a student"])

# ---------------- Tab 1: performance ----------------
with tab_perf:
    st.subheader("Results on the independent test set")

    metrics = results["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", metrics["accuracy"])
    col2.metric("F1 (macro)", metrics["f1_macro"])
    col3.metric("Precision (macro)", metrics["precision_macro"])
    col4.metric("ROC-AUC", metrics["roc_auc"])

    with st.expander("All metrics"):
        st.table(pd.Series(metrics, name="value"))

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=results["labels"], yticklabels=results["labels"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.caption(
        "Low and High are predicted well. The Medium class is the hardest, and "
        "confusing Low with High is very rare."
    )

# ---------------- Tab 2: test one student ----------------
with tab_predict:
    st.subheader("Enter a student's information")

    with st.form("student_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 16, 40, 21)
            year = st.slider("Year of study", 1, 5, 2)
            sleep = st.slider("Sleep hours per night", 3.0, 10.0, 7.0, 0.1)
            study = st.slider("Study hours per week", 0.0, 50.0, 20.0, 0.5)
            gpa = st.slider("GPA (4.0 scale)", 1.0, 4.0, 2.6, 0.01)
            social = st.slider("Social media hours per day", 0.0, 10.0, 6.0, 0.1)
            screen = st.slider("Screen time hours per day", 0.0, 14.0, 8.5, 0.1)

        with c2:
            stress = st.slider("Financial stress (1-10)", 1.0, 10.0, 6.0, 0.1)
            support = st.slider("Family support (1-10)", 1.0, 10.0, 6.5, 0.1)
            attendance = st.slider("Attendance rate (%)", 0.0, 100.0, 85.0, 0.5)
            activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"], index=1)
            diet = st.selectbox("Diet quality", ["Poor", "Average", "Good"], index=1)
            job = st.selectbox("Part-time job", ["No", "Yes"])
            history = st.selectbox("Previous mental health history", ["No", "Yes"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        student = pd.DataFrame([{
            "Age": age,
            "Gender": "Female",  # required column, but the model ignores it
            "Year_of_Study": year,
            "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study,
            "GPA": gpa,
            "Social_Media_Hours": social,
            "Screen_Time_Hours": screen,
            "Physical_Activity": activity,
            "Financial_Stress": stress,
            "Family_Support": support,
            "Attendance_Rate": attendance,
            "Part_Time_Job": job,
            "Diet_Quality": diet,
            "Previous_Mental_Health_History": history,
        }])

        try:
            prediction = model.predict(student)[0]
            probabilities = pd.Series(model.predict_proba(student)[0], index=model.classes_)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

        st.success(f"Predicted risk level: **{prediction}**")
        st.bar_chart(probabilities.reindex(LABELS))
        st.caption("Bars show the model's estimated probability for each level.")