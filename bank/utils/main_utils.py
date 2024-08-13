import yaml
import pandas as pd
import numpy as np
import os
import dill
import sys
from bank.exception import BankException
from bank.logger import logging




# This function reads a YAML file from the specified file path and returns the data
# as a dictionary. If an error occurs while reading the file, it raises a BankException
# with the error message and the system details.

def read_yaml_file(file_path: str) -> dict:
    try:
        # Open the YAML file in binary mode
        with open(file_path, "rb") as yaml_file:
            # Use the safe_load function from the yaml module to load the data from the
            # file and return it as a dictionary.
            return yaml.safe_load(yaml_file)

    except Exception as e:
        # If an error occurs, raise a BankException with the error message and the system
        # details. The error message is obtained from the exception object and the system
        # details are obtained from the sys module.
        raise BankException(e, sys)
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise BankException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise BankException(e, sys)
