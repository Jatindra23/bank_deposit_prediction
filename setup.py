from setuptools import find_packages, setup
from typing import List
from bank.exception import BankException
from bank.logger import logging
import sys


def get_requirements() -> List[str]:
    requirements_list: List[str] = []

    try:
        logging.info("trying to open requirementrs.txt file")
        with open("requirements.txt", "r") as f:
            requirements_list = f.readlines()

    except Exception as e:
        logging.exception(f"an error occcured {e}")
        raise BankException(e, sys)

    # Strip newline characters and remove '-e .' from the list
    requirements_list = [
        requirement.strip()
        for requirement in requirements_list
        if requirement.strip() != "-e ."
    ]

    return requirements_list


setup(
    name="bank_term_deposit_prediction",
    version="0.0.0.1",
    author="Jatindra Paul",
    author_email="jatindrapaul7@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
