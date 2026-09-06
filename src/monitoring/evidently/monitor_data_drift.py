import os
import sys
from datetime import datetime

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.database.connection import SessionLocal
from src.database.models import ModelPredictionLog
from src.exception import CustomException
from src.logger import logging
from src.monitoring.evidently.data_loader import (
    get_current_dataframe,
    get_reference_train_dataframe,
)
from src.utils import load_config

logging.info("Loading the config.yaml file")
config = load_config()


def plotting_data_drift(session, table_name, reference_path, model_version, date_format) -> None:
    """
    Generate a data drift report comparing input feature distributions between
    recent production data and the reference (baseline training) data.

    Production 'input_features' (stored as a JSON list containing one dict) is
    flattened into individual feature columns to match the reference dataframe's
    structure. The 'truth_label' column is aligned to string dtype on both sides
    for consistent categorical comparison.

    Args:
        session: SQLAlchemy sessionmaker factory used to create a DB session.
        table_name: SQLAlchemy ORM model/table representing prediction logs.
        reference_path (str): Path to the baseline training data (.npy) used as reference.
        model_version (str): Model version to filter predictions by (e.g., "v1").
        date_format (str): Datetime format string used for naming the saved report.

    Returns:
        None. Saves an HTML data drift report to disk.

    Raises:
        CustomException: If data loading, drift computation, or report saving fails.
    """
    try:
        logging.info(f"Fetching current and reference dataframes for model_version={model_version}")

        current_dataframe_all_columns = get_current_dataframe(
            session=session,
            table_name=table_name,
            model_version=model_version
        )
        reference_dataframe_all_columns = get_reference_train_dataframe(
            reference_path=reference_path
        )

        logging.info("Selecting 'input_features' and 'truth_label' columns from current data")

        current_dataframe_selected_columns = current_dataframe_all_columns[["input_features", "truth_label"]].copy()

        logging.info("Unwrapping 'input_features' list and flattening into individual feature columns")

        current_dataframe_selected_columns['input_features'] = current_dataframe_selected_columns['input_features'].apply(lambda x: x[0])

        expanded_features = pd.json_normalize(current_dataframe_selected_columns['input_features'])

        logging.info("Concatenating unwrapped input_features to the truth_label")

        current_dataframe_final = pd.concat(
            [expanded_features, current_dataframe_selected_columns.drop(columns=['input_features'])],
            axis=1
        )

        logging.info("Aligning reference 'truth_label' dtype to string to match current data (varchar)")

        # Align dtype to string: prevents mismatched categories (e.g., "1" vs 1.0) from inflating drift score
        reference_dataframe_all_columns['truth_label'] = reference_dataframe_all_columns['truth_label'].astype('int64').astype(str)

        numerical_columns = [
            'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year'
        ]

        categorical_columns = [
            'Month', 'Day', 'Weekday',
            'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm',
            'RainToday', 'truth_label'
        ]

        data_definition = DataDefinition(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )

        logging.info("Wrapping current and reference dataframes into Evidently Dataset objects")

        current_dataset = Dataset.from_pandas(current_dataframe_final, data_definition=data_definition)
        reference_dataset = Dataset.from_pandas(reference_dataframe_all_columns, data_definition=data_definition)

        logging.info("Running DataDriftPreset report")

        report = Report([DataDriftPreset()])
        my_eval = report.run(current_dataset, reference_dataset)

        report_dir = os.path.join("reports", f"{model_version}")
        os.makedirs(report_dir, exist_ok=True)

        path_to_save = os.path.join(report_dir, f"report_{datetime.now().strftime(date_format)}.html")

        my_eval.save_html(path_to_save)

        logging.info(f"Data drift report saved successfully at {path_to_save}")

    except Exception as e:
        logging.exception("An error occurred while generating data drift report")
        raise CustomException(e, sys)


if __name__ == "__main__":

    logging.info("Starting the model monitoring script")

    data_drift_baseline_path = os.path.join("src/monitoring/baselines", f"data_drift_baseline_{config['model_version']}.npy")

    plotting_data_drift(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path=data_drift_baseline_path,
        model_version=config['model_version'],
        date_format=config['date_format']
    )
