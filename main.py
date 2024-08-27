from bank.exception import BankException
import sys, os
import pandas as pd
from bank.logger import logging
from bank.pipeline.training_pipeline import TrainPipeline
from bank.configuration.mongodb_connection import MongoDBClient
from bank.constant.database import DATABASE_NAME
from bank.ml.model.estimator import ModelResolver, TargetValueMapping
from bank.constant.training_pipeline import SAVED_MODEL_DIR
from bank.utils.main_utils import load_object, read_yaml_file
from fastapi import FastAPI, File, UploadFile, Response
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from bank.constant.application import APP_HOST, APP_PORT
from bank.exception import BankException


if __name__ == "__main__":

    try:

        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()

    except Exception as e:
        raise BankException(e, sys)
