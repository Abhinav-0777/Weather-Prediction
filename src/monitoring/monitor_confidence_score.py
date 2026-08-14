import os
import sys
from datetime import datetime
from src.logger import logging
from src.utils import load_config
from evidently.metrics import ValueDrift
from src.exception import CustomException
from evidently import (
    Report,
    DataDefinition,
    Dataset
)
from src.services.database import(
    SessionLocal,
    ModelPredictionLog
)
from src.monitoring.data_loader import (
    get_confidence_score_current_data,
    get_confidence_score_reference_data
)


config = load_config()

def plotting_confidence_score(session, table_name, reference_path, model_version, date_format):

    try:
        logging.info(f"Starting confidence_score drift monitoring for model_version={model_version}")

        current_dataframe_final = get_confidence_score_current_data(session=session,
                                                                    table_name=table_name,
                                                                    model_version=model_version)
        logging.info(f"Fetched current data: {current_dataframe_final.shape[0]} rows")

        reference_dataframe_final = get_confidence_score_reference_data(reference_path=reference_path)
        logging.info(f"Loaded reference data: {reference_dataframe_final.shape[0]} rows")
      
        data_definition = DataDefinition(
            numerical_columns = ['confidence_score']
        )

        current_dataset = Dataset.from_pandas(current_dataframe_final, data_definition = data_definition)
        reference_dataset = Dataset.from_pandas(reference_dataframe_final, data_definition = data_definition)

        logging.info("Running ValueDrift report on confidence_score")
        report = Report([ValueDrift(column='confidence_score')])

        report_result = report.run(current_dataset, reference_dataset)

        report_dir = os.path.join("reports", f"{model_version}")
        os.makedirs(report_dir, exist_ok=True)

        path_to_save = os.path.join(report_dir, f"report_{datetime.now().strftime(date_format)}.html")

        report_result.save_html(path_to_save)
        logging.info(f"Drift report saved successfully at {path_to_save}")

        return report_result.dict()


    except Exception as e:
        logging.exception("An error has occurred while monitoring the confidence_score metric")
        raise CustomException(e,sys)


if __name__ == "__main__" :

    logging.info("Running confidence_score monitoring script as standalone")

    result = plotting_confidence_score(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path="src/monitoring/baselines/confidence_score_baseline_v1.csv",
        model_version=config['model_version'],
        date_format=config['date_format']
    )