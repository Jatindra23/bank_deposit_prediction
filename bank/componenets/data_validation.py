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
