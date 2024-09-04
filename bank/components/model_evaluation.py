from bank.exception import BankException
from bank.logger import logging

from bank.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from bank.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig

import os, sys

from bank.ml.metric.classification_metric import get_classification_score

from bank.ml.model.estimator import BankModel
from bank.utils.main_utils import load_object, save_object, write_yaml_file
from bank.ml.model.estimator import ModelResolver
from bank.constant.training_pipeline import TARGET_COLUMN
from bank.ml.model.estimator import TargetValueMapping
import pandas as pd
import numpy as np


class ModelEvaluation:

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:

            self.model_evaluation_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise BankException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:

            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df, test_df])
            logging.info(f"number of rows and columns in df: {df.shape}")
            df = df.drop_duplicates()
            logging.info(
                f"number of rows and columns after removing duplicates: {df.shape}"
            )

            y_true = df[TARGET_COLUMN]

            y_true.replace(TargetValueMapping().to_dict(), inplace=True)

            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            train_model_file_path = (
                self.model_trainer_artifact.trained_model_object_file_path
            )
            model_resolver = ModelResolver()

            is_model_accepted = True

            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    trained_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None,
                )
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_classification_score(
                y_true=y_true, y_pred=y_trained_pred
            )

            latest_metric = get_classification_score(
                y_true=y_true, y_pred=y_latest_pred
            )

            improved_accuracy = trained_metric.f1_score - latest_metric.f1_score
            if self.model_evaluation_config.change_threshold < improved_accuracy:
                is_model_accepted = True

            else:
                is_model_accepted = False

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                trained_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric,
            )

            model_evaluate_report = model_evaluation_artifact.__dict__

            model_eval_report = self.convert_to_basic_types(data=model_evaluate_report)

            # save the report
            model_eval_dir_path = os.path.dirname(
                self.model_evaluation_config.report_file_path
            )
            os.makedirs(model_eval_dir_path, exist_ok=True)
            write_yaml_file(
                self.model_evaluation_config.report_file_path, model_eval_report
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise BankException(e, sys)

    def convert_to_basic_types(self, data):
        """
        Recursively converts complex data types to their basic equivalents.

        This function is designed to handle nested data structures, such as dictionaries and lists,
        and convert them to a format that can be easily serialized or processed.

        Args:
            data: The data to be converted. Can be a dictionary, list, numpy scalar, numpy array, or custom object.
                - If data is a dictionary, its values will be recursively converted.
                - If data is a list, its elements will be recursively converted.
                - If data is a numpy scalar, it will be converted to a Python scalar.
                - If data is a numpy array, it will be converted to a Python list.
                - If data is a custom object with a __dict__ attribute, its attributes will be recursively converted.

        Returns:
            The converted data in its basic type.
                - Dictionaries will be converted to dictionaries with basic types as values.
                - Lists will be converted to lists with basic types as elements.
                - Numpy scalars will be converted to Python scalars.
                - Numpy arrays will be converted to Python lists.
                - Custom objects will be converted to dictionaries with basic types as values.

        """

        if isinstance(data, dict):
            return {k: self.convert_to_basic_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_to_basic_types(item) for item in data]
        elif isinstance(data, np.generic):  # Handles numpy scalars
            return data.item()
        elif isinstance(data, np.ndarray):  # Handles numpy arrays
            return data.tolist()
        elif hasattr(data, "__dict__"):  # Handles custom objects with __dict__
            return self.convert_to_basic_types(data.__dict__)
        else:
            return data
