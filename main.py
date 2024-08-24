from bank.exception import BankException
import sys
from bank.logger import logging
from bank.pipeline.training_pipeline import TrainPipeline
from bank.configuration.mongodb_connection import MongoDBClient
from bank.constant.database import DATABASE_NAME





if __name__ == "__main__":

    try:
        training_pipeline = TrainPipeline()
        
        training_pipeline.run_pipeline()

    except Exception as e:
        raise BankException(e, sys)


