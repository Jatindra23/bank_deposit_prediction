from distutils import dir_util
from bank.constant.training_pipeline import SCHEMA_FILE_PATH
from bank.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from bank.entity.config_entity import DataValidationConfig
from bank.exception import BankException
from bank.logger import logging
from bank.utils.main_utils import read_yaml_file, write_yaml_file
import pandas as pd
import os
import sys
import numpy as np
import json
import warnings

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.report import Report
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab

warnings.filterwarnings("ignore")


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            np.random.seed(42)

        except Exception as e:
            raise BankException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Reqired number of columns : {number_of_columns}")
            logging.info(f"Data frame has columns :{len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
                return True
            return False

        except Exception as e:
            raise BankException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:

        try:

            numerical_columns = self._schema_config["numerical_columns"]
            numeric_columns = [list(cols.keys())[0] for cols in numerical_columns]
            dataframe_columns = dataframe.select_dtypes(include="int64").columns

            numerical_column_present = True
            missing_numericla_columns = []

            for num_column in numeric_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numericla_columns.append(num_column)

            logging.info(f"Missing numerical columns: {missing_numericla_columns}")

            return numerical_column_present

        except Exception as e:
            raise BankException(e, sys)

    def is_categorical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:

            categorical_columns = self._schema_config["categorical_columns"]
            categorical_col = [cols for cols in categorical_columns]

            dataframe_columns = dataframe.select_dtypes(include="object").columns

            categorical_column_present = True
            missing_categorical_columns = []

            for cat_column in categorical_col:
                if cat_column not in dataframe_columns:
                    categorical_column_present = False
                    missing_categorical_columns.append(cat_column)

            logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return categorical_column_present

        except Exception as e:
            raise BankException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads a CSV file from the given file path and returns the data as a pandas DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The contents of the CSV file as a DataFrame.

        Raises:
            BankException: If an error occurs while reading the CSV file.
        """
        try:

            return pd.read_csv(file_path)

        except Exception as e:
            raise BankException(e, sys)

    def remove_duplicates(
        self, test_data: pd.DataFrame, train_data: pd.DataFrame
    ) -> pd.DataFrame:
        try:
            test_data.drop_duplicates(ignore_index=True, inplace=True)
            train_data.drop_duplicates(ignore_index=True, inplace=True)

            return test_data, train_data
        except Exception as e:
            raise BankException(e, sys)

    

    def get_data_drift_report_page(self, current_df, base_df):

        try:

            data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
            data_drift_dashboard.calculate(current_df, base_df)
            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.data_drift_report_page_file_path
                ),
                exist_ok=True,
            )
            html_report = data_drift_dashboard.save(
                self.data_validation_config.data_drift_report_page_file_path
            )

            return html_report
        except Exception as e:
            raise BankException(e, sys)

    def detect_dataset_drift(self, base_df, current_df) -> bool:
        try:

            data_drift_profile = Profile(sections=[DataDriftProfileSection()])

            data_drift_profile.calculate(base_df, current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report,
            )

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"][
                "n_drifted_features"
            ]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            logging.info(f"The drift Status is: {drift_status}")
            return drift_status

        except Exception as e:
            raise BankException(e, sys)

    def initiate_data_validation(self) -> DataValidationConfig:

        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # reading data from train and test file
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # removing duplicates
            train_dataframe, test_dataframe = self.remove_duplicates(
                test_data=test_dataframe, train_data=train_dataframe
            )
            logging.info(
                f"the shape of train dataframe: {train_dataframe.shape} and test dataframe: {test_dataframe.shape}"
            )

            # validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message} Train dataframe doesn't contain all the columns.\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = (
                    f"{error_message} test dataframe doen't contain all the columns.\n"
                )

            num_col_status = self.is_train_numerical_column_exist = (
                self.is_numerical_column_exist(dataframe=train_dataframe)
            )
            if not num_col_status:
                error_message = f"{error_message} Train dataframe doesn't contain all the numerical columns.\n"

            num_col_status = self.is_train_numerical_column_exist = (
                self.is_numerical_column_exist(dataframe=test_dataframe)
            )
            if not num_col_status:
                error_message = f"{error_message} test dataframe doen't contain all the numerical columns.\n"

            # validate categorical columns
            status = self.is_categorical_column_exist(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message} Train dataframe doesn't contain all the categorical columns.\n"

            status = self.is_categorical_column_exist(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message} test dataframe doen't contain all the categorical columns.\n"

            # validate dataset drift

            if len(error_message) > 0:
                raise Exception(error_message)

            drift_status = self.detect_dataset_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )
            drift_report_page = self.get_data_drift_report_page(
                current_df=test_dataframe, base_df=train_dataframe
            )

            if drift_status:
                status = False
            else:
                status = True

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                data_drift_report_page_file_path=drift_report_page,
            )

            logging.info(f"Data_validation_artifact: {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            raise BankException(e, sys)
