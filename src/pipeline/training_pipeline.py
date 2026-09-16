from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import model_evaluation
from src.components.model_trainer import ModelTrainer


def run_ingestion(**kwargs):
    DataIngestion_obj = DataIngestion()
    train_data_path, test_data_path = DataIngestion_obj.initiate_data_ingestion()
    kwargs['ti'].xcom_push(key='train_data_path', value=train_data_path)
    kwargs['ti'].xcom_push(key='test_data_path', value=test_data_path)


def run_transformation(**kwargs):
    DataTransformation_obj = DataTransformation()
    train_data_path = kwargs['ti'].xcom_pull(task_ids='ingest_data', key='train_data_path')
    test_data_path = kwargs['ti'].xcom_pull(task_ids='ingest_data', key='test_data_path')
    transformed_train_array_path, transformed_test_array_path, preprocessor_path = DataTransformation_obj.initiate_data_transformation(
        train_data_path=train_data_path,
        test_data_path=test_data_path
    )
    kwargs['ti'].xcom_push(key='transformed_train_array_path', value=transformed_train_array_path)
    kwargs['ti'].xcom_push(key='transformed_test_array_path', value=transformed_test_array_path)
    kwargs['ti'].xcom_push(key='preprocessor_path', value=preprocessor_path)


def run_training(**kwargs):
    ModelTrainer_obj = ModelTrainer()
    transformed_train_array_path = kwargs['ti'].xcom_pull(task_ids='transform_data', key='transformed_train_array_path')
    transformed_test_array_path = kwargs['ti'].xcom_pull(task_ids='transform_data', key='transformed_test_array_path')
    preprocessor_path = kwargs['ti'].xcom_pull(task_ids='transform_data', key='preprocessor_path')
    ModelTrainer_obj.initiate_model_trainer(
        transformed_train_array_path=transformed_train_array_path,
        transformed_test_array_path=transformed_test_array_path,
        preprocessor_path=preprocessor_path
    )


def run_evaluation(**kwargs):
    transformed_test_array_path = kwargs['ti'].xcom_pull(task_ids='transform_data', key='transformed_test_array_path')
    model_evaluation(transformed_test_array_path)
