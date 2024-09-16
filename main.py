from bank.exception import BankException
import sys, os
import pandas as pd
from bank.logger import logging
from bank.pipeline.training_pipeline import TrainPipeline
from bank.configuration.mongodb_connection import MongoDBClient
from bank.constant.database import DATABASE_NAME
from bank.ml.model.estimator import ModelResolver, TargetValueMapping
from bank.ml.metric.classification_metric import get_classification_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from bank.constant.training_pipeline import SAVED_MODEL_DIR
from bank.utils.main_utils import load_object, read_yaml_file
from fastapi import FastAPI, File, UploadFile, Response
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from bank.constant.application import APP_HOST, APP_PORT
from bank.entity.artifact_entity import DataValidationArtifact
import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi import Request
from typing import Optional
from bank.utils2 import BankInputData, BankClassifier


# ...


# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/", tags=["authentication"])
# async def index():
#     return RedirectResponse(url="/docs")


# @app.get("/train")
# async def train():
#     try:
#         training_pipeline = TrainPipeline()
#         if training_pipeline.is_pipeline_running:
#             return Response("Training Pipelin is already running.")

#         training_pipeline.run_pipeline()
#         return Response("Training pipeline has been successfully initiated.")
#     except Exception as e:
#         return Response(f"Error Occured! {e}")


# @app.get("/predict")
# async def predict():
#     try:
#         t_df = pd.read_csv(
#             r"C:\Users\JOY\Desktop\bank_term_deposit_prediction\bank_clean1.csv"
#         )
#         t_df.drop_duplicates(inplace=True)

#         # Get the features and target
#         df = t_df.iloc[:, :-1]
#         df_target = t_df.iloc[:, -1]

#         # Check for missing values
#         if df.isnull().any().any() or df_target.isnull().any():
#             return Response("Data contains missing values.", status_code=400)

#         # Replace target values
#         target_mapping = TargetValueMapping().to_dict()
#         target_values = df_target.replace(target_mapping)

#         model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
#         if not model_resolver.is_model_exists():
#             return Response("Model is not available !")

#         best_model_path = model_resolver.get_best_model_path()
#         best_model = load_object(best_model_path)
#         y_predict = best_model.predict(df)

#         # Convert predictions to pandas Series and replace target values
#         y_predict_series = pd.Series(y_predict)
#         y_predict_mapped = y_predict_series.replace(target_mapping)

#         # Check for NaN values in predictions and target values
#         if target_values.isnull().any() or y_predict_mapped.isnull().any():
#             return Response(
#                 "Target values or predictions contain NaN values.", status_code=400
#             )

#         # Calculate F1 score
#         if len(target_values) == len(y_predict_mapped):
#             classif = f1_score(target_values, y_predict_mapped, average="weighted")
#         else:
#             classif = "Length mismatch between target_values and y_predict"

#         # Create DataFrame for results
#         result = pd.DataFrame({"Actual": target_values, "Predicted": y_predict_mapped})

#         result_str = result.to_string(index=False)
#         if isinstance(classif, str):
#             result_str += f"\nF1_Score: {classif}"
#         else:
#             result_str += f"\nF1_Score: {classif:.4f}"

#         report = classification_report(target_values, y_predict_mapped)
#         print("Classification Report:\n", report)

#         confuse_matrix = confusion_matrix(target_values, y_predict_mapped)

#         return Response(
#             f"Prediction results:\n{report}\n Confusion Matrix:\n\n{confuse_matrix}",
#             status_code=200,
#         )

#     except Exception as e:
#         return Response(f"An error occurred: {str(e)}", status_code=500)


# def main():
#     try:
#         training_pipeline = TrainPipeline()
#         training_pipeline.run_pipeline()

#     except Exception as e:
#         raise BankException(e, sys)


# if __name__ == "__main__":

#     uvicorn.run(app, host="127.0.0.1", port=APP_PORT)


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.age: Optional[int] = None
        self.balance: Optional[int] = None
        self.day: Optional[int] = None
        self.duration: Optional[int] = None
        self.campaign: Optional[int] = None
        self.previous: Optional[int] = None
        self.job: Optional[str] = None
        self.marital: Optional[str] = None
        self.education: Optional[str] = None
        self.default: Optional[str] = None
        self.housing: Optional[str] = None
        self.default: Optional[str] = None
        self.loan: Optional[str] = None
        self.contact: Optional[str] = None
        self.month: Optional[str] = None

    async def get_bank_data(self):
        form = await self.request.form()
        self.age = form.get("age")
        self.balance = form.get("balance")
        self.day = form.get("day")
        self.duration = form.get("duration")
        self.campaign = form.get("campaign")
        self.previous = form.get("previous")
        self.job = form.get("job")
        self.marital = form.get("marital")
        self.education = form.get("education")
        self.default = form.get("default")
        self.housing = form.get("housing")
        self.loan = form.get("loan")
        self.contact = form.get("contact")
        self.month = form.get("month")


@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
        "index.html", {"request": request, "context": "Rendering"}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        logging.info("Form submitted")
        form = DataForm(request)
        await form.get_bank_data()
        # logging.info(f"Form data: age={form.age}, balance={form.balance}, day={form.day}")

        bank_data = BankInputData(
            age=form.age,
            balance=form.balance,
            day=form.day,
            duration=form.duration,
            campaign=form.campaign,
            previous=form.previous,
            job=form.job,
            marital=form.marital,
            education=form.education,
            default=form.default,
            housing=form.housing,
            loan=form.loan,
            contact=form.contact,
            month=form.month,
        )

        bank_df = bank_data.get_bank_input_data_frame()

        model_predictor = BankClassifier()

        value = model_predictor.predict(dataframe=bank_df)[0]
        logging.info(f"Prediction result: {value}")

        if value == 1:
            status = "Bank Deposit Policy taken"
        else:
            status = "Bank Deposit Policy Not taken "

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        print(f"Error occurred: {e}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "context": f"Error: {e}"}
        )


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)  # Update port if needed

# "127.0.0.1"
