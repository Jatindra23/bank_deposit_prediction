import os
from bank.exception import BankException
import sys


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
            y_hat = self.model.predict(x_transform)
            return y_hat

        except Exception as e:
            raise BankException(e, sys)
