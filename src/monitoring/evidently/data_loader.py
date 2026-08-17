import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import func

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


def get_current_dataframe(session, table_name, model_version) -> pd.DataFrame:
    """
    Fetch label-safe production predictions from the database for a rolling
    7-day window (offset by a 2-day label-arrival delay), filtered by model version.

    Args:
        session: SQLAlchemy sessionmaker factory used to create a DB session.
        table_name: SQLAlchemy ORM model/table representing prediction logs.
        model_version (str): Model version to filter predictions by (e.g., "v1").

    Returns:
        pd.DataFrame: DataFrame containing predictions with non-null truth labels,
        within the label-safe rolling window, for the given model version.

    Raises:
        CustomException: If the database query or session handling fails.
    """

    db = None

    try:
        logging.info("Creating a new session to fetch current_data")
        db = session()

        db_date = datetime.now(UTC)
        window_start = db_date.date() - timedelta(days=8)
        window_end = db_date.date() - timedelta(days=2)

        logging.info(f"Querying predictions for model_version={model_version} "
                     f"total number of days={(window_end-window_start).days + 1} "
                     f"between {window_start} and {window_end}")

        current_data = (
            db.query(table_name)
            .filter(func.date(table_name.timestamp) >= window_start)
            .filter(func.date(table_name.timestamp) <= window_end)
            .filter(table_name.truth_label.isnot(None))
            .filter(table_name.model_version == model_version)
        )

        current_dataframe = pd.read_sql(current_data.statement, db.bind)

        logging.info(f"Fetched {len(current_dataframe)} rows from the database "
                     f"for model_version={model_version}")

        return current_dataframe

    except Exception as e:
        logging.exception("An error occurred while fetching current_dataframe")
        raise CustomException(e, sys)

    finally:
        if db:
            logging.info("Closing the database session")
            db.close()


def get_reference_evaluation_dataframe(reference_path) -> pd.DataFrame:
    """
    Load the reference (baseline) evaluation dataframe from a stored CSV file
    — used for performance/target drift checks.

    Args:
        reference_path (str): Path to the saved baseline evaluation CSV
        (containing predictions and ground truth for the test set).

    Returns:
        pd.DataFrame: Reference dataframe used as the baseline distribution
        for performance and target drift comparison.

    Raises:
        CustomException: If the file is missing or cannot be read.
    """
    try:
        logging.info(f"Loading reference evaluation dataframe from {reference_path}")

        reference_dataframe = pd.read_csv(reference_path)

        logging.info(f"Loaded reference evaluation dataframe with "
                     f"{reference_dataframe.shape[0]} rows and {reference_dataframe.shape[1]} columns")

        return reference_dataframe

    except Exception as e:
        logging.exception("An error occurred while generating reference_evaluation_dataframe")
        raise CustomException(e, sys)


def get_reference_train_dataframe(reference_path) -> pd.DataFrame:
    """
    Load the reference (baseline) training dataframe used for data drift checks.

    Args:
        reference_path (str): Path to the serialized (pickled) training array.

    Returns:
        pd.DataFrame: Reconstructed training dataframe with named columns,
        used as the baseline distribution for data drift comparison.

    Raises:
        CustomException: If loading or reconstructing the dataframe fails.
    """
    try:
        logging.info(f"Loading reference training array from {reference_path}")

        reference_array = load_object(reference_path)

        reference_dataframe = pd.DataFrame(
            reference_array,
            columns=[
                'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month',
                'Day', 'Weekday', 'Location', 'WindGustDir', 'WindDir9am',
                'WindDir3pm', 'RainToday', 'truth_label'
            ]
        )

        logging.info(f"Reconstructed reference training dataframe with "
                     f"{reference_dataframe.shape[0]} rows and {reference_dataframe.shape[1]} columns")

        return reference_dataframe

    except Exception as e:
        logging.exception("An error occurred while generating reference_train_dataframe")
        raise CustomException(e, sys)


def get_confidence_score_current_data(session, table_name, model_version) -> pd.DataFrame:

    db = None

    try:
        logging.info("Creating a new session to fetch confidence_score")
        db = session()

        confidence_score_query = (
            db.query(table_name.confidence_score, table_name.timestamp)
            .filter(table_name.model_version == model_version)
            .filter(table_name.client_type == "authorized_client")
        )

        confidence_score_dataframe = pd.read_sql(confidence_score_query.statement, db.bind)

        logging.info(f"Fetched {len(confidence_score_dataframe)} confidence_score rows "
                     f"for model_version={model_version}")

        return confidence_score_dataframe

    except Exception as e:
        logging.exception("An error has occurred while getting confidence_score metric from Supabase")
        raise CustomException(e, sys)

    finally:
        if db:
            logging.info("Closing the database session")
            db.close()


def get_confidence_score_reference_data(reference_path) -> pd.DataFrame:

    try:
        logging.info(f"Loading reference confidence_score data from {reference_path}")

        confidence_score_reference_data = pd.read_csv(reference_path)

        logging.info(f"Loaded reference data: {confidence_score_reference_data.shape[0]} rows")

        return confidence_score_reference_data

    except Exception as e:
        logging.exception("An error has occurred while loading reference confidence_score data")
        raise CustomException(e, sys)
