import os
import sys
import time
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, load_object, save_object


@dataclass
class ModelTrainerConfig :

    """Configures the ModelTrainer class and sets up a path for saving the trained model.
    """

    trained_model_file_path : str = os.path.join("artifacts","model.pkl")

class ModelTrainer :

    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, transformed_train_array_path, transformed_test_array_path, preprocessor_path) :

        """
        Trains multiple classification models and selects the best one
        based on accuracy score.

        Steps:
        1. Split train and test arrays into features and target.
        2. Train multiple classification models.
        3. Evaluate models using accuracy score.
        4. Select the best performing model.
        5. Save the trained model.

        Args:
            train_array (np.array): Transformed training dataset
            test_array (np.array): Transformed testing dataset

        Returns:
            float: Accuracy score of the best model on test data
        """

        start_time = time.time()

        try :

            logging.info("Model Training has started")

            logging.debug(f"Loading transformed train array from: {transformed_train_array_path}")
            logging.debug(f"Loading transformed test array from: {transformed_test_array_path}")
            logging.debug(f"Preprocessor path received: {preprocessor_path}")

            train_array = load_object(transformed_train_array_path)
            test_array = load_object(transformed_test_array_path)

            logging.info(f"Loaded train array with shape {train_array.shape} and test array with shape {test_array.shape}")

            logging.info("Splitting the train and test arrays")

            X_train, y_train, X_test, y_test = [
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ]

            logging.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.debug(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            models = {

                'Logistic Regression' : LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
                'Decision Trees' : DecisionTreeClassifier(class_weight='balanced', random_state=42),
                'KNN' : KNeighborsClassifier(),
                'Random Forest' : RandomForestClassifier(class_weight='balanced', random_state=42),
                'XGBoost' : XGBClassifier(scale_pos_weight=3.46, device='cuda', random_state=42),
                'CatBoost' : CatBoostClassifier(verbose=False, scale_pos_weight=3.46, task_type='GPU', random_state=42),
                'AdaBoost' : AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'), random_state=42),
                'GradientBoosting' : GradientBoostingClassifier(random_state=42)

            }

            logging.info(f"Initialized {len(models)} candidate models: {list(models.keys())}")

            params = {

                    "Logistic Regression": {
                        'C': [0.001,0.01,0.1,1,10,100],
                        'solver': ['lbfgs']
                    },

                    "Decision Trees": {
                        'criterion': ['gini', 'entropy', 'log_loss'],
                        'max_depth': [None,5,10,20]
                    },

                    "KNN": {
                        'n_neighbors': [3,5,7,9],
                        'weights': ['uniform','distance'],
                        'metric': ['minkowski','euclidean','manhattan']
                    },

                    "Random Forest": {
                        'n_estimators': [8,16,32,64,128,256],
                        'criterion': ['gini','entropy','log_loss']
                    },

                    "XGBoost": {
                        'learning_rate': [.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256],
                        'eval_metric': ['logloss','aucpr']
                    },

                    "CatBoost": {
                        'depth': [6,8,10],
                        'learning_rate': [0.01,0.05,0.1],
                        'iterations': [30,50,100,200]
                    },

                    "AdaBoost": {
                        'learning_rate': [.1,.01,0.5,.001],
                        'n_estimators': [8,16,32,64,128,256]
                    },

                    "GradientBoosting": {
                        'learning_rate': [.1,.01,.05],
                        'subsample': [0.7,0.75,0.8],
                        'n_estimators': [32,64,128]
                    }

            }

            mlflow.set_experiment("weather_prediction_classification")
            logging.info("MLflow experiment set to 'weather_prediction_classification'")

            with mlflow.start_run(run_name="weather_model_parent") as parent_run:

                logging.info(f"Started MLflow parent run with run_id: {parent_run.info.run_id}")

                model_training_start = time.time()

                model_report : dict = evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models=models,
                    params = params
                )

                model_training_duration = time.time() - model_training_start
                logging.info(f"Completed training and evaluation of all models in {model_training_duration:.2f} seconds")
                logging.info(f"Model report (F2 scores): {model_report}")

                best_model_f2_score = max(model_report.values())

                best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_f2_score)]

                best_model = models[best_model_name]

                logging.info(f"Best model selected: {best_model_name} with F2 score: {best_model_f2_score:.4f}")

                if best_model_f2_score < 0.6 :
                    logging.warning(f"Best model F2 score ({best_model_f2_score:.4f}) is below the acceptable threshold of 0.6")
                    raise CustomException("No good model exists currently")

                logging.info(f"Saving the best model to: {self.model_trainer_config.trained_model_file_path}")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                logging.info("Best model saved successfully")

                mlflow.log_param('best_model_name', best_model_name)
                mlflow.log_metric('best_model_f2_score', best_model_f2_score)

                logging.info("Logging best model to MLflow model registry as 'weather_prediction_classifier'")

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name="weather_prediction_classifier"
                )

                logging.info("Model successfully registered in MLflow")

                total_duration = time.time() - start_time
                logging.info(f"Model training pipeline completed in {total_duration:.2f} seconds")
                logging.info(f"The best model f2_score is: {best_model_f2_score}")

                return best_model_f2_score

        except Exception as e :
            logging.exception("An error has occurred during model training")
            raise CustomException(e,sys)


if __name__ == "__main__" :
    obj = ModelTrainer()
    obj.initiate_model_trainer()
