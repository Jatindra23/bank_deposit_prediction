from distutils import dir_util
from bank.constant.training_pipeline import SCHEMA_FILE_PATH
from bank.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from bank.entity.config_entity import DataValidationConfig
from bank.exception import BankException
from bank.logger import logging
from bank.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
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
            self.data_ingestion_aartifact = data_ingestion_artifact
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
        

    def is_numerical_column_exist(self,dataframe:pd.DataFrame)-> bool:
        try:

            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numericla_columns = []

            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_poresent = False
                    missing_numericla_columns.append(num_column)

            logging.info(f"Missing numerical columns: {missing_numericla_columns}")

            return numerical_column_present

        except Exception as e:
            raise BankException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
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
            raise BankException (e,sys)
        


    def detect_dataset_drift(self, base_df, current_df, threshold: float = 0.05) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dit = ks_2samp(d1,d2)

        except Exception as e:
            raise BankException (e,sys)
        
