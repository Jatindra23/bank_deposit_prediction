from bank.utils.main_utils import load_numpy_array_data
from bank.exception import BankException
from bank.logger import logging
from bank.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from bank.entity.config_entity import ModelTrainerConfig
import os, sys

from xgboost import XGBClassifier
from bank.ml.metric.classification_metric import get_classification_score
from bank.ml.model.estimator import BankModel
from bank.utils.main_utils import save_object, load_object


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise BankException(e, sys)

    def perform_hyper_parameter_tuning(self):
        try:
            pass

        except Exception as e:
            raise BankException(e, sys)

    def train_model(self, x_train, y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)
            return xgb_clf
        except Exception as e:
            raise BankException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            # loading training arrary testing array

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )

            if (
                classification_train_metric.f1_score
                <= self.model_trainer_config.expected_accuracy
            ):
                raise BankException(
                    "Model is not good enough. Please try again with different parameters",
                    sys,
                )

            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )

            # Overfitting and Underfitting
            diff = abs(
                classification_train_metric.f1_score
                - classification_test_metric.f1_score
            )

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise BankException(
                    "Model is either overfitting or underfitting. Please try again with other mdoel"
                )

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir_path, exist_ok=True)
            bank_model = BankModel(preprocessor=preprocessor, model=model)
            save_object(
                self.model_trainer_config.trained_model_file_path, obj=bank_model
            )

            # model_trainer_artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_object_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise BankException(e, sys)
