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
from bank.entity.artifact_entity import DataValidationArtifact
import uvicorn

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train():
    try:
        training_pipeline = TrainPipeline()
        if training_pipeline.is_pipeline_running:
            return Response("Training Pipelin is already running.")

        training_pipeline.run_pipeline()
        return Response("Training pipeline has been successfully initiated.")
    except Exception as e:
        return Response(f"Error Occured! {e}")


@app.get("/predict")
async def predict():
    try:
        # get data from the csv file
        # convert it into data frame
        # data_validation_artifact = DataValidationArtifact()
        # testing_file = data_validation_artifact.valid_test_file_path
        # df = pd.read_csv(testing_file)

        t_df = pd.read_csv(
            r"C:\Users\JOY\Desktop\bank_term_deposit_prediction\bank_clean.csv"
        )
        t_df.drop_duplicates(inplace=True)

        df = t_df.iloc[800:825, :]

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available !")

        best_model_path = model_resolver.get_best_model_path()
        best_model = load_object(best_model_path)
        y_pred = best_model.predict(df)
        df["predicted_column"] = y_pred
        target_value_mapping = TargetValueMapping()
        df["predicted_column"].replace(
            target_value_mapping.reverse_mapping(), inplace=True
        )

        # get the output as you want
        # Simplify the data
        simplified_data = [
            {"y": row["y"], "predicted_column": row["predicted_column"]}
            for _, row in df.iterrows()
        ]
        # for record in simplified_data:
        #     print(f"Actual: {record['y']}, Predicted: {record['predicted_column']}")
        # predictions = df.to_dict(orient="records")
        return Response(f"this is the result {simplified_data}")
    except Exception as e:
        return Response(f"An error occurred: {str(e)}", status_code=500)


def main():
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()

    except Exception as e:
        raise BankException(e, sys)


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=APP_PORT)
