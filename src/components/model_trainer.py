import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig :

    """Configures the ModelTrainer class and sets up a path for saving the trained model.
    """

    trained_model_file_path : str = os.path.join("artifacts","model.pkl")

class ModelTrainer :

    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path) :

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


        try :
            logging.info("Model Training has started")

            logging.info("Splitting the train and test arrays")

            X_train, y_train, X_test, y_test = [
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ]

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

            model_report : dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params = params
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                raise CustomException("No good model exists currently")

            logging.info("Saving the best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            best_model_prediction = best_model.predict(X_test)

            best_model_f2_score = fbeta_score(y_test, best_model_prediction, beta=2)

            logging.info(f"The best model f2_score is: {best_model_f2_score}")

            return best_model_f2_score

        except Exception as e :
            logging.exception("An error has occurred")
            raise CustomException(e,sys)

if __name__ == "__main__" :
    obj = ModelTrainer()
    obj.initiate_model_trainer()
