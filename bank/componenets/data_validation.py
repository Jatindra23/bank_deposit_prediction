from distutils import dir_util
from bank.constant.training_pipeline import SCHEMA_FILE_PATH
from bank.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from bank.entity.config_entity import DataValidationConfig
from bank.exception import BankException
from bank.logger import logging
from bank.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp, chi2_contingency
import pandas as pd
import os
import sys


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
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numericla_columns = []

            for num_column in numerical_columns:
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
            dataframe_columns = dataframe.columns

            categorical_column_present = True
            missing_categorical_columns = []

            for cat_column in self._schema_config["categorical_columns"]:
                if cat_column not in dataframe.columns:
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

    def detect_dataset_drift(
        self, base_df, current_df, threshold: float = 0.05
    ) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False

                else:
                    is_found = True
                    status = False

                report.update(
                    {
                        column: {
                            "p_value": float(is_same_dist.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )

            for column in base_df.select_dtypes(include=["object"]).columns:
                d1 = base_df[column]
                d2 = current_df[column]

                # Create contingency table
                contingency_table = pd.crosstab(d1, d2)
                chi2, p_value, dof, ex = chi2_contingency(
                    contingency_table, correction=False
                )

                if p_value < threshold:
                    is_found = False

                else:
                    is_found = True
                    status = False

                report.update(
                    {column: {"p_value": float(p_value), "drift_status": not is_found}}
                )

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

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

            # validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message} Train dataframe doesn't contain all the columns.\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = (
                    f"{error_message} test dataframe doen't contain all the columns.\n"
                )

            if len(error_message) > 0:
                raise Exception(error_message)

            status = self.detect_dataset_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data_validation_artifact: {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            raise BankException(e, sys)
