import pandas as pd
import sys
from os import path, makedirs
import shutil
import logging
from typing import List

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


def handle_duplicate_values_drop(data : pd.DataFrame, subset : List = None) -> pd.DataFrame:
    # Check dataset to know how many duplicate values exist
    # Find duplicate values
        # keep='first' (default): Marks duplicates as True, except for the first occurrence.
        # keep='last': Marks duplicates as True, except for the last occurrence.
        # keep=False: Marks all duplicates (including the first and last) as True.
    data_duplicated = data.duplicated(keep=False, subset=subset)
    logging.info(f"Dataset has {data.shape[0]} rows before handling duplicate values.\nTop 10 of duplicate values are (Totally {data_duplicated.sum()} rows - including all duplicates, but from each group first one will remain and others will be removed):\n{data[data_duplicated]}")

    # Remove duplicate values
    # Subset is list of column names which we want to participate in the duplicate recognition
    # If it is None, all column values of a row should be the same as other's to consider as duplicates
    data = data.drop_duplicates(subset=subset)

    # Check dataset rows after removing duplicate rows
    logging.info(f"Dataset has {data.shape[0]} rows before handling duplicate values.")

    return data


def main():
    # Start logging
    config_logging()

    # Check the argument passed to the program
    # Note that the first argument is always the name of the program
    if len(sys.argv) < 2:
        logging.error("This program needs at least one parameter as the dataset path!")
        sys.exit(0)
    else:
        match len(sys.argv):
            case 2:
                dataset_path = sys.argv[1]
                duplicate_cols_subset = None
            case 3:
                dataset_path = sys.argv[1]
                # This parameter should pass to the program in comma seperated format e.g., "First Name,Last Name" (Obviously column names are case-sebsitive)
                duplicate_cols_subset = sys.argv[2].split(",")

    # Load the dataset
    original_data = load_data(dataset_path)
    # If the dataset is not valid
    if original_data.empty:
        return
    
    # Create a folder for cleaned datasets
    dataset_dir = path.dirname(dataset_path)
    cleaned_data_dir = path.join(dataset_dir, "../", "cleaned_data_handle_duplicate_values")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(cleaned_data_dir):
        shutil.rmtree(cleaned_data_dir)
    # Create the folder
    makedirs(cleaned_data_dir, exist_ok=True)

    # Handle duplicate values using drop method
    data = original_data.copy()
    data_cleaned_drop = handle_duplicate_values_drop(data, duplicate_cols_subset)
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_drop.empty:
        data_cleaned_drop.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_drop.csv"), index=False)


if __name__ == "__main__":
    main()