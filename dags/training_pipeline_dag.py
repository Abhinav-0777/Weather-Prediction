from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.training_pipeline import (
    run_evaluation,
    run_ingestion,
    run_training,
    run_transformation,
)

with DAG("model_pipeline", start_date=datetime(2026,9,1), schedule=None, catchup=False) as dag:
    t1 = PythonOperator(task_id="ingest_data", python_callable=run_ingestion)
    t2 = PythonOperator(task_id="transform_data", python_callable=run_transformation, retries=3, retry_delay=timedelta(minutes=5))
    t3 = PythonOperator(task_id="train_model", python_callable=run_training, retries=4, retry_delay=timedelta(minutes=10))
    t4 = PythonOperator(task_id="evaluate_model", python_callable=run_evaluation, retries=3, retry_delay=timedelta(minutes=5))

    t1 >> t2 >> t3 >> t4
