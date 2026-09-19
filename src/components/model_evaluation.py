import json
import os
import sys
import time

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

from src.exception import CustomException
from src.logger import logging
from src.utils import load_config, load_object

config = load_config()

def model_evaluation(transformed_test_path) -> dict:

    """Evaluates the trained model on the test dataset and saves performance metrics.
       This function loads the transformed test array and the trained model,
       generates predictions, computes evaluation metrics (F2-score, confusion
       matrix, classification report), and saves them to a JSON file for tracking.

    Raises:
        CustomException: If any error occurs during model loading, prediction,
            or metrics computation/saving.

    Returns:
        dict: A dictionary containing the F2-score, confusion matrix, and
            classification report of the model's performance on the test set.
    """

    start_time = time.time()

    try :

        logging.info("Model evaluation has started")

        logging.debug(f"Loading transformed test array from: {transformed_test_path}")

        test_arr = load_object(transformed_test_path)

        logging.info(f"Successfully loaded the transformed_test_array object with shape {test_arr.shape}")

        X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

        logging.debug(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        logging.info("Successfully divided the test array into features and target columns")

        logging.info(f"Loading the pickled model file from: {config['model_path']}")

        model_obj = load_object(config['model_path'])

        logging.info(f"Successfully loaded model of type: {type(model_obj).__name__}")

        logging.info("Predicting on the data")

        prediction_start = time.time()
        y_pred = model_obj.predict(X_test)
        prediction_duration = time.time() - prediction_start

        logging.info(f"Prediction completed in {prediction_duration:.4f} seconds for {len(y_pred)} samples")

        logging.info("Calculating the metrics")

        f2 = fbeta_score(y_test, y_pred, beta=2)
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, target_names=['Not Rain','Rain'], output_dict=True)

        logging.info(f"F2 SCORE: {f2}")
        logging.info(f"\nCONFUSION MATRIX:\n {cm}")
        logging.info(f"\nCLASSIFICATION REPORT:\n {cr}")

        metrics = {
            "f2_score": float(f2),
            "confusion_matrix": cm,
            "classification_report": cr
        }

        metrics_path = config['metrics_path']

        logging.debug(f"Writing metrics to: {metrics_path}")

        with open(metrics_path, "w") as f :
            json.dump(metrics, f, indent=4)

        logging.info("Saved the metrics to the artifacts folder")

        mlflow.set_experiment('weather_prediction_classification')
        logging.info("MLflow experiment set to 'weather_prediction_classification'")

        client = MlflowClient()
        versions = client.search_model_versions("name='weather_prediction_classifier'")

        if not versions :
            logging.warning("No registered versions found for 'weather_prediction_classifier'; evaluation run will not be linked to a training run")
            training_run_id = None
        else :
            latest_version = max(versions, key=lambda v: int(v.version))
            training_run_id = latest_version.run_id
            logging.info(f"Found latest registered model version: {latest_version.version}, linked to training run_id: {training_run_id}")

        with mlflow.start_run(run_name="final_model_evaluation") as eval_run:

            logging.info(f"Started MLflow evaluation run with run_id: {eval_run.info.run_id}")

            if training_run_id :
                mlflow.set_tag("mlflow.parentRunId", training_run_id)
                logging.info(f"Tagged evaluation run as a child of training run_id: {training_run_id}")

            mlflow.log_metric("f2_score", f2)
            mlflow.log_artifact(metrics_path)

            logging.info("Logged f2_score metric and metrics.json artifact to MLflow")

        df_eval = pd.DataFrame({'truth_label': y_test, 'predicted_output': y_pred})

        df = df_eval.astype('int64')

        baseline_path = os.path.join(
            config['baseline_dir'],
            f"predictions_baseline_{config['model_version']}.csv"
        )

        logging.info(f"Saving the new baseline to '{baseline_path}' file")

        df.to_csv(baseline_path, header=True, index=None)

        logging.info(f"Baseline saved successfully with {len(df)} rows")

        total_duration = time.time() - start_time
        logging.info(f"Model evaluation completed in {total_duration:.2f} seconds")

        return metrics


    except Exception as e :
        logging.exception("An error has occurred during model evaluation")
        raise CustomException(e,sys)


if __name__ == "__main__":
    model_evaluation("artifacts/transformed_test_array.npy")
