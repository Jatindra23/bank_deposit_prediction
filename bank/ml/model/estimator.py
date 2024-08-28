import os
from bank.exception import BankException
import sys
import numpy as np

from bank.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME


class TargetValueMapping:

    def __init__(self):
        self.no: int = 0
        self.yes: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class BankModel:

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model

        except Exception as e:
            raise BankException(e, sys)

    def predict(self, x):
        try:

            x_transform = self.preprocessor.transform(x)
            testing_arr = np.c_[x_transform]
            y_hat = self.model.predict(testing_arr)
            return y_hat

        except Exception as e:
            raise BankException(e, sys)


class ModelResolver:

    def __init__(self, model_dir: str = SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise BankException(e, sys)

    def get_best_model_path(self) -> str:
        """
        Retrieves the path of the best model based on the latest timestamp in the model directory.

        Args:
            None

        Returns:
            str: The path of the best model.
        """
        try:

            timestamps = list(map(int, os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)

            latest_model_path = os.path.join(
                self.model_dir, f"{latest_timestamp}", MODEL_FILE_NAME
            )

            return latest_model_path

        except Exception as e:
            raise BankException(e, sys)

    def is_model_exists(self) -> bool:
        """
        Checks if a model exists in the model directory.

         Args:
           None

         Returns:
           bool: True if the model exists, False otherwise.
        """
        try:

            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False

            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True

        except Exception as e:
            raise BankException(e, sys)
