import pandas as pd
from bank.logger import logging
from bank.exception import BankException
import sys
import json
from bank.configuration import mongo_client



def dump_file_to_mongodb_collection(file_path:str,databse_name:str,collection_name:str)-> None:

    try:
        logging.info("started to read file")
        df = pd.read_csv(file_path)
        df.reset_index(drop= True)

        # the function json.loads()is used to get the output as python dictionary, later it converted into list
        json_records = list(json.loads(df.T.to_json( )).values())

        mongo_client[databse_name][collection_name].insert_many(json_records)

    except Exception as e:
        logging.exception(f"An error has occured: {e}")
        raise BankException(e,sys)
