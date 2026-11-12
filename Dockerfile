FROM apache/airflow:2.10.5-python3.12

# Separate virtual environment for the ML libraries
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN python -m venv /home/airflow/ml-venv \
    && /home/airflow/ml-venv/bin/pip install --no-cache-dir -r /tmp/requirements-airflow.txt