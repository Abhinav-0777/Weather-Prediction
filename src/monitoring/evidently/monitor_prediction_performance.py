import os
import sys
from datetime import datetime

from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset

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


def plotting_predictions_performance(session, table_name, reference_path, model_version, date_format) -> None:
    """
    Generate a classification performance report comparing model predictions
    against ground-truth labels, for both current (production) and reference
    (baseline evaluation) data.

    Both 'truth_label' and 'predicted_output' are cast to int64 to ensure
    compatibility with Evidently's classification metrics.

    Args:
        ...

    Returns:
        None. Saves an HTML performance report to disk.

    Raises:
        CustomException: If data loading, metric computation, or report saving fails.
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

        logging.info("Selecting 'truth_label' and 'predicted_output' columns from current and reference data")

        current_dataframe_selected_columns = current_dataframe_all_columns[['truth_label', 'predicted_output']]
        reference_dataframe_selected_columns = reference_dataframe_all_columns[['truth_label', 'predicted_output']]

        logging.info("Casting truth_label and predicted_output to int64 on both dataframes for ClassificationPreset compatibility")

        # Cast to int64: Evidently's BinaryClassification expects an integer pos_label by default
        current_dataframe_final = current_dataframe_selected_columns.astype({'truth_label': 'int64', 'predicted_output': 'int64'})
        reference_dataframe_final = reference_dataframe_selected_columns.astype({'truth_label': 'int64', 'predicted_output': 'int64'})

        logging.info("Defining the classification schema for Evidently (target and prediction columns)")

        data_definition = DataDefinition(
            classification=[BinaryClassification(
                target='truth_label',
                prediction_labels='predicted_output'
            )]
        )

        logging.info("Wrapping current and reference dataframes into Evidently Dataset objects")

        current_dataset = Dataset.from_pandas(current_dataframe_final, data_definition=data_definition)
        reference_dataset = Dataset.from_pandas(reference_dataframe_final, data_definition=data_definition)

        logging.info("Running the ClassificationPreset report")

        report = Report([ClassificationPreset()])
        my_eval = report.run(current_dataset, reference_dataset)

        report_dir = os.path.join("reports", f"{model_version}")
        os.makedirs(report_dir, exist_ok=True)

        path_to_save = os.path.join(report_dir, f"report_{datetime.now().strftime(date_format)}.html")

        my_eval.save_html(path_to_save)

        logging.info(f"Performance report saved successfully at {path_to_save}")

    except Exception as e:
        logging.exception("An error occurred while generating performance report")
        raise CustomException(e, sys)


if __name__ == "__main__":

    logging.info("Starting the model monitoring script")

    prediction_baseline_path = os.path.join("src/monitoring/baselines", f"predictions_baseline_{config['model_version']}.csv")

    plotting_predictions_performance(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path=prediction_baseline_path,
        model_version=config['model_version'],
        date_format=config['date_format']
    )

    logging.info("Model monitoring script completed")
