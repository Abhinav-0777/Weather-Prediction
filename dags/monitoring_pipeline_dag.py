from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.monitoring_pipeline import (
    run_monitor_confidence_score,
    run_monitor_data_drift,
    run_monitor_prediction_drift,
    run_monitor_prediction_performance,
)

with DAG("monitoring_pipeline", start_date=datetime(2026,9,1), schedule=None, catchup=False) as dag:
    t1 = PythonOperator(task_id="monitor_confidence_score", python_callable=run_monitor_confidence_score)
    t2 = PythonOperator(task_id="monitor_data_drift", python_callable=run_monitor_data_drift)
    t3 = PythonOperator(task_id="monitor_prediction_drift", python_callable=run_monitor_prediction_drift)
    t4 = PythonOperator(task_id="monitor_prediction_performance", python_callable=run_monitor_prediction_performance)

    [t1, t2, t3, t4]
