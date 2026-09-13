from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.training_pipeline import (
    run_ingestion,
    run_training,
    run_transformation,
)

with DAG("model_pipeline", start_date=datetime(2026,9,1), schedule_interval=None, catchup=False) as dag:
    t1 = PythonOperator(task_id="ingest_data", python_callable=run_ingestion)
    t2 = PythonOperator(task_id="transform_data", python_callable=run_transformation)
    t3 = PythonOperator(task_id="train_model", python_callable=run_training)

    t1 >> t2 >> t3
