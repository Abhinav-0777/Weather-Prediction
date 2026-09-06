import os
import sys
from datetime import datetime

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift

from src.database.connection import SessionLocal
from src.database.models import ModelPredictionLog
from src.exception import CustomException
from src.logger import logging
from src.monitoring.evidently.data_loader import (
    get_current_dataframe,
    get_reference_evaluation_dataframe,
)
from src.utils import load_config

logging.info("Loading the config.yaml file")
config = load_config()


def plotting_predictions_drift(session, table_name, reference_path, model_version, date_format) -> None:
    """
    Generate a prediction drift report comparing the 'predicted_output' column
    between recent production data and the reference (baseline evaluation) data.

    The column is treated as categorical (binary label), and both current and
    reference values are aligned to string dtype for consistent comparison.

    Args:
        session: SQLAlchemy sessionmaker factory used to create a DB session.
        table_name: SQLAlchemy ORM model/table representing prediction logs.
        reference_path (str): Path to the baseline evaluation CSV used as reference.
        model_version (str): Model version to filter predictions by (e.g., "v1").
        date_format (str): Datetime format string used for naming the saved report.

    Returns:
        None. Saves an HTML drift report to disk.

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
        reference_dataframe_all_columns = get_reference_evaluation_dataframe(
            reference_path=reference_path
        )

        logging.info("Selecting 'predicted_output' column from current and reference dataframes")

        current_dataframe_final = current_dataframe_all_columns[['predicted_output']]
        reference_dataframe_selected_columns = reference_dataframe_all_columns[['predicted_output']]

        logging.info("Aligning reference 'predicted_output' dtype to string to match current data (varchar)")

        # Align dtype to string: prevents mismatched categories (e.g., "0" vs 0) from inflating drift score
        reference_dataframe_final = reference_dataframe_selected_columns.astype(str)

        data_definition = DataDefinition(
            categorical_columns=['predicted_output']
        )

        logging.info("Wrapping current and reference dataframes into Evidently Dataset objects")

        current_dataset = Dataset.from_pandas(current_dataframe_final, data_definition=data_definition)
        reference_dataset = Dataset.from_pandas(reference_dataframe_final, data_definition=data_definition)

        logging.info("Running ValueDrift metric on 'predicted_output'")

        report = Report([ValueDrift(column='predicted_output')])
        my_eval = report.run(current_dataset, reference_dataset)

        report_dir = os.path.join("reports", f"{model_version}")
        os.makedirs(report_dir, exist_ok=True)

        path_to_save = os.path.join(report_dir, f"report_{datetime.now().strftime(date_format)}.html")

        my_eval.save_html(path_to_save)

        logging.info(f"Prediction drift report saved successfully at {path_to_save}")

    except Exception as e:
        logging.exception("An error occurred while generating prediction drift report")
        raise CustomException(e, sys)


if __name__ == "__main__":

    prediction_baseline_path = os.path.join("src/monitoring/baselines", f"predictions_baseline_{config['model_version']}.csv")

    plotting_predictions_drift(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path=prediction_baseline_path,
        model_version=config["model_version"],
        date_format=config["date_format"]
    )
