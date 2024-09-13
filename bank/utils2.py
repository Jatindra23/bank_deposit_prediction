import pandas as pd
from bank.logger import logging
from bank.exception import BankException
import sys
import json

from bank.configuration.mongodb_connection import MongoDBClient as mongo_client
from bank.ml.model.estimator import BankModel
from bank.entity.artifact_entity import ModelTrainerArtifact
from bank.ml.model.estimator import ModelResolver
from bank.utils.main_utils import load_object
import os, sys
from pandas import DataFrame
from bank.constant.training_pipeline import SAVED_MODEL_DIR


def dump_file_to_mongodb_collection(
    file_path: str, databse_name: str, collection_name: str
) -> None:

    try:
        logging.info("started to read file")
        df = pd.read_csv(file_path)
        df.reset_index(drop=True)

        # the function json.loads()is used to get the output as python dictionary, later it converted into list
        json_records = list(json.loads(df.T.to_json()).values())

        mongo_client[databse_name][collection_name].insert_many(json_records)

    except Exception as e:
        logging.exception(f"An error has occured: {e}")
        raise BankException(e, sys)


class BankInputData:
    def __init__(
        self,
        age,
        balance,
        day,
        duration,
        campaign,
        previous,
        job,
        marital,
        education,
        default,
        housing,
        loan,
        contact,
        month,
    ):
        """
        Bank Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.age = age
            self.balance = balance
            self.day = day
            self.duration = duration
            self.campaign = campaign
            self.previous = previous
            self.job = job
            self.marital = marital
            self.education = education
            self.default = default
            self.housing = housing
            self.loan = loan
            self.contact = contact
            self.month = month

        except Exception as e:
            raise BankException(e, sys) from e

    def get_bank_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from BankInputData class input
        """
        try:

            bank_input_dict = self.get_bank_data_as_dict()
            return DataFrame(bank_input_dict)

        except Exception as e:
            raise BankException(e, sys)

    def get_bank_data_as_dict(self):
        """
        This function returns a dictionary from USvisaData class input
        """
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "age": [self.age],
                "balance": [self.balance],
                "day": [self.day],
                "duration": [self.duration],
                "campaign": [self.campaign],
                "previous": [self.previous],
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "default": [self.default],
                "housing": [self.housing],
                "loan": [self.loan],
                "contact": [self.contact],
                "month": [self.month],
            }

            logging.info("Created usvisa data dict")

            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class")

            return input_data

        except Exception as e:
            raise BankException(e, sys)


class BankClassifier:

    def predict(self, dataframe) -> str:
        """
        This is the method of BankClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of BankClassifier class")

            model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)

            if not model_resolver.is_model_exists():
                print("model_is_not_available")

            best_model_path = model_resolver.get_best_model_path()
            best_model = load_object(best_model_path)

            prediction = best_model.predict(dataframe)

            return prediction

        except Exception as e:
            raise BankException(e, sys)
