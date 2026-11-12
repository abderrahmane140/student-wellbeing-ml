from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PYTHON = "/home/airflow/ml-venv/bin/python"
PROJECT = "/opt/project"


def notify_failure(context):
    """Called by Airflow when a task fails (visible in the task logs)."""
    task = context["task_instance"].task_id
    print(f"ALERT: task '{task}' failed.")


default_args = {
    "retries": 1,                          
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=10),
    "on_failure_callback": notify_failure,
}

def make_task(task_id, step_name):
    """One task = one command from src/pipeline_steps.py."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT} && {PYTHON} -m src.pipeline_steps {step_name}",
    )


with DAG(
    dag_id="student_risk_pipeline",
    description="Student mental health risk: prepare, preprocess, train, evaluate",
    start_date=datetime(2026, 11, 1),
    schedule="@weekly",                   
    catchup=False,                       
    default_args=default_args,
    tags=["student", "ml"],
) as dag:

    prepare = make_task("prepare_data", "prepare")
    preprocess = make_task("check_preprocessing", "preprocess")
    train = make_task("train_model", "train")
    evaluate = make_task("evaluate_model", "evaluate")

    prepare >> preprocess >> train >> evaluate