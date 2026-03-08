import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
import os
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig :
    transformed_data_obj_file_path : str = os.path.join("artifacts","data_transformed.pkl")


class DataTransformation() :

    def __init__(self) :
        self.DataTransformationConfig = DataTransformationConfig()

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
                ('num_pipeline', num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            
            logging.info("Successfully created the data_transformer object")

            return preprocessor

        except Exception as e:
            logging.info("An error has occurred")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path) :
        
        logging.info("Data transformation has started")
        
        try :

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Successfully read the train and test data")
            
            for data in [train_df, test_df] :

                data['Date'] = pd.to_datetime(data['Date'])

                data['Year'] = data['Date'].dt.year
                data['Month'] = data['Date'].dt.month
                data['Day'] = data['Date'].dt.day
                data['Weekday'] = data['Date'].dt.weekday
                
                data = data.drop(columns = 'Date')


            target_column_name = 'RainTomorrow'

            logging.info("Dividing the train and test set into input and target features")
 
            input_feature_train_df = train_df.drop(columns=target_column_name)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df[target_column_name]

            label_encoder_object = LabelEncoder()
            target_feature_train_arr = label_encoder_object.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder_object.transform(target_feature_test_df)
            print(target_feature_train_arr.shape)

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
            print(input_feature_train_arr.shape)       
            print(input_feature_train_arr)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            
            save_object (
                self.DataTransformationConfig.transformed_data_obj_file_path,
                preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.DataTransformationConfig.transformed_data_obj_file_path
            )
        

        except Exception as e:
            logging.info("An error has occurred")
            raise CustomException(e,sys)

