import pandas as pd
import sys
from os import path, makedirs
import shutil
import logging
from typing import List
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from enum import Enum


class CategoricalEncodingMethod(Enum):
    LABEL_ENCODING = 1
    ONEHOT_ENCODING = 2
    HASHING = 3


def config_logging():
    # This function configs logging and prepares it for logging process
    # Note that logging creates logs from every operation has been done in the program which is the more flexible, durable, and powerful than only using print
    # Whenever you want a new logging, just delete the existing one!
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("all_operations.log")
    ]
    )


def load_data(file_path : str) -> pd.DataFrame:
    # Open csv file and load it into a dataframe
    try:
        data = pd.read_csv(file_path)
    except:
        logging.error("The path is invalid!")
        return pd.DataFrame()

    # Return the first 5 rows of the dataset
    logging.info(f"\n{data.head()}")

    return data


def get_observing_columns(data : pd.DataFrame, columns_subset : List) -> List:
    # Prepare observing columns
    # Strip whitespaces
    if columns_subset: columns_subset = [col.strip() for col in columns_subset]
    try:
        # If columns_subset only has categorical columns is valid
        categorical_columns = data.select_dtypes(exclude="number").columns
        # If columns_subset is not None and one of its columns does not exist in categorical columns
        if columns_subset and not all(col in categorical_columns for col in columns_subset):
            logging.error("The columns subset contains numeric columns!")
            return []
        else:
            # If there is a valid subset, it is considered as the observing columns otherwise all categorical columns are considered
            observing_columns = columns_subset if columns_subset else categorical_columns
    except:
        logging.error("The columns subset is not valid!")
        return []

    return observing_columns 



def main():
    # Start logging
    config_logging()

    # Check the argument passed to the program
    # Note that the first argument is always the name of the program
    if len(sys.argv) < 4:
        logging.error("This program needs at least three parameter as the dataset path!")
        sys.exit(0)
    else:
        match len(sys.argv):
            case 4:
                dataset_path = sys.argv[1]
                # This parameter should pass to the program in comma seperated format e.g., "First Name,Last Name" (Obviously column names are case-sebsitive)
                columns_subset = sys.argv[2].split(",")
                columns_scaling_method = sys.argv[3].split(",")
                apply_l2normalization = False
            case 5:
                dataset_path = sys.argv[1]
                # This parameter should pass to the program in comma seperated format e.g., "First Name,Last Name" (Obviously column names are case-sebsitive)
                columns_subset = sys.argv[2].split(",")
                columns_scaling_method = sys.argv[3].split(",")
                apply_l2normalization = sys.argv[4].lower() == "true"


    # Load the dataset
    original_data = load_data(dataset_path)
    # If the dataset is not valid
    if original_data.empty:
        return
    
    # Create a folder for cleaned datasets
    dataset_dir = path.dirname(dataset_path)
    output_dir = path.join(dataset_dir, "../", "output_encode_categorical")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the folder
    makedirs(output_dir, exist_ok=True)



if __name__ == "__main__":
    main()