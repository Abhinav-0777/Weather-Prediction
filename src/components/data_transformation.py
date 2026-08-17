import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig :
    transformed_data_obj_file_path : str = os.path.join("artifacts","preprocessing_object.pkl")
    transformed_train_array_path : str = os.path.join("artifacts","transformed_train_array.npy")
    transformed_test_array_path : str = os.path.join("artifacts","transformed_test_array.npy")

class DataTransformation :

    def __init__(self) :
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns) :

        """Creates and returns a preprocessing pipeline for numerical
           and categorical features.

        Returns:
            _sklearn.compose.ColumnTransformer_ : A preprocessing object that applies numerical and categorical
                                                  transformation pipelines to the dataset.
        """

        logging.info("Data transformation started")

        try :

            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder())
                ]
            )

            logging.info("Numerical and categorical columns pipeline created")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info("Successfully created the data_transformer object")

            return preprocessor

        except Exception as e:
            logging.exception("An error has occurred")
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path) :

        """
        Initiates the data transformation process.

        This function loads the training and testing datasets, performs feature
        engineering and preprocessing using a transformation pipeline, and
        converts the data into a format suitable for model training.

        Steps performed:
        1. Load train and test datasets from the given file paths.
        2. Separate input features and target variable.
        3. Apply preprocessing pipeline (scaling, encoding, etc.).
        4. Transform both train and test datasets.
        5. Save the preprocessing object for future use.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            tuple: Transformed training array, transformed testing array,
                and path to the saved preprocessing object.

        Raises:
            CustomException: If any error occurs during the transformation process.
        """

        logging.info("Data transformation has started")

        try :

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully read the train and test data")

            train_df = train_df.dropna(subset=['RainTomorrow'])
            test_df = test_df.dropna(subset=['RainTomorrow'])

            logging.info("Successfully dropped the rows with NaN target values")

            for data in [train_df, test_df] :

                data['Date'] = pd.to_datetime(data['Date'])

                data['Year'] = data['Date'].dt.year
                data['Month'] = data['Date'].dt.month
                data['Day'] = data['Date'].dt.day
                data['Weekday'] = data['Date'].dt.weekday

                data.drop(columns = 'Date', inplace=True)

            target_column_name = 'RainTomorrow'

            logging.info("Dividing the train and test set into input and target features")

            input_feature_train_df = train_df.drop(columns=target_column_name)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df[target_column_name]

            label_encoder_object = LabelEncoder()
            target_feature_train_arr = label_encoder_object.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder_object.transform(target_feature_test_df)

            numerical_columns = input_feature_train_df.select_dtypes(include=np.number).columns
            categorical_columns = input_feature_train_df.select_dtypes(include=['object','category']).columns

            logging.info("Obtaining the preprocessor object.... ")

            preprocessing_obj = self.get_data_transformer_object(
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns
            )

            logging.info("Applying the preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Successfully applied the preprocessing object")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saved the transformed train array to the artifacts folder")

            save_object(self.data_transformation_config.transformed_train_array_path,
                        train_arr)

            logging.info("Saved the transformed test array to the artifacts folder")

            save_object(self.data_transformation_config.transformed_test_array_path,
                        test_arr)

            logging.info("Saving the data")

            save_object (
                self.data_transformation_config.transformed_data_obj_file_path,
                preprocessing_obj
            )

            logging.info("Data saved successfully into preprocessing_object.pkl")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.transformed_data_obj_file_path
            )


        except Exception as e:
            logging.exception("An error has occurred")
            raise CustomException(e,sys)

