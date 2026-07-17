import os 
import sys
import json
import pandas as pd
from src.logger import logging
from src.utils import load_object
from src.exception import CustomException
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score


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

    try :
    
        test_arr = load_object(transformed_test_path)

        logging.info("Successfully loaded the transformed_test_array object")

        X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

        logging.info("Successfully divided the test array into features and target columns")

        logging.info("Loading the pickled model file")

        model_path = os.path.join("artifacts","model.pkl")
        model_obj = load_object(model_path)
        
        logging.info("Predicting on the data")

        y_pred = model_obj.predict(X_test)

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

        metrics_path = os.path.join("artifacts","metrics.json")

        with open(metrics_path, "w") as f :
            json.dump(metrics, f, indent=4)

        logging.info("Saved the metrics to the artifacts folder")

        df = pd.DataFrame({'truth_label': y_test, 'predicted_output': y_pred})

        return df.astype('int64')


    except Exception as e :
        logging.exception("An error has occurred")
        raise CustomException(e,sys)


if __name__ == "__main__":
    model_evaluation("artifacts/transformed_test_array.npy")
