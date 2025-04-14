import pandas as pd
import numpy as np
from enum import Enum
import sys
from os import path, makedirs
from typing import List, Dict, Tuple
import shutil
import logging

class DetectOutlierMethod(Enum):
    IQR = 1
    ZSCORE = 2
    ISOLATION_FOREST = 3
    LOCAL_OUTLIER_FACTOR = 4
    ONECLASS_SVM = 5

class HandleOutlierMethod(Enum):
    DROP = 1
    REPLACE_WITH_MEDIAN = 2
    CAP_WITH_BOUNDARIES = 3

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
    try:
        # If columns_subset only has numeric columns is valid
        numeric_columns = data.select_dtypes(include="number").columns
        # If columns_subset is not None and one of its columns does not exist in numeric columns
        if columns_subset and not all(col in numeric_columns for col in columns_subset):
            logging.error("The columns subset contains non-numeric columns!")
            return []
        else:
            # If there is a valid subset, it is considered as the observing columns otherwise all nemuric columns are considered
            observing_columns = columns_subset if columns_subset else numeric_columns
    except:
        logging.error("The columns subset is not valid!")
        return []

    return observing_columns    


def detect_outliers(data : pd.DataFrame, detect_outlier_method : DetectOutlierMethod, columns_subset : List = None) -> Tuple:
    # Check if column_subset is valid
    observing_columns = get_observing_columns(data, columns_subset)
    if len(observing_columns) == 0: return dict(), dict()

    outliers = {}
    boundries = {}
    # Check detecting method and run the following block
    match detect_outlier_method:
        case DetectOutlierMethod.IQR:
            for col in observing_columns:
                # Calculate quantiles 1, 3
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                # Extract outliers based on IQR method
                outliers[col] = data.loc[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)].index.to_list()
                boundries[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        case DetectOutlierMethod.ZSCORE:
            for col in observing_columns:
                mean = data[col].mean()
                std = data[col].std()
                
                # Extract outliers based on Z-Score method
                # Z-Score = (data - mean)/std  -->  to be inlier  -->  -3 < Z-Score < 3
                outliers[col] = data.loc[(data[col] < mean - 3 * std) | (data[col] > mean + 3 * std)].index.to_list()
                boundries[col] = (mean - 3 * std, mean + 3 * std)

        case DetectOutlierMethod.ISOLATION_FOREST:
            pass
        case DetectOutlierMethod.LOCAL_OUTLIER_FACTOR:
            pass
        case DetectOutlierMethod.ONECLASS_SVM:
            pass
    
    # Output of the function is a Tuple consists of oulier indexes dict and boundries on inliers dict
    return outliers, boundries


def handle_outliers(data : pd.DataFrame, handle_outlier_method : HandleOutlierMethod, outliers : Dict = {}, boundries : Dict = {}) -> pd.DataFrame :
    # If the outlier dict is empty, the output is the original data
    if len(outliers) == 0: return data

    # First unpack the values of the outlier dict and then union all of them in a set (to eliminate duplicate indexes)
    all_drop_indexes = set().union(*outliers.values())
    
    # Check dataset to know how many duplicate values exist
    # Find duplicate values
    logging.info(f"Dataset has {data.shape[0]} rows before handling outliers values.\nTop 10 of rows containing outliers are (Totally {len(all_drop_indexes)} rows):\n{data.iloc[list(all_drop_indexes)]}")

    match handle_outlier_method:
        case HandleOutlierMethod.DROP:
            # Drop all outliers
            data = data.drop(all_drop_indexes)
        case HandleOutlierMethod.REPLACE_WITH_MEDIAN:
            for col in outliers.keys():
                # For each column which has outliers, all the outliers replace with Median of that column
                data.loc[outliers[col], col] = data[col].median()
        case HandleOutlierMethod.CAP_WITH_BOUNDARIES:
            for col in outliers.keys():
                # For each column which has outliers, all the outliers cap (clip) with boundry values of that column
                lower, upper = boundries[col]
                print(lower, upper)
                data.loc[outliers[col], col] = data.loc[outliers[col], col].clip(lower, upper)

    # Check dataset rows after removing duplicate rows
    logging.info(f"Dataset has {data.shape[0]} rows after handling outliers.")

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
                columns_subset = None
            case 3:
                dataset_path = sys.argv[1]
                # This parameter should pass to the program in comma seperated format e.g., "First Name,Last Name" (Obviously column names are case-sebsitive)
                if sys.argv[2] == "None":
                    columns_subset = None
                else:
                    columns_subset = sys.argv[2].split(",")

    # Load the dataset
    original_data = load_data(dataset_path)
    # If the dataset is not valid
    if original_data.empty:
        return
    
    # Create a folder for cleaned datasets
    dataset_dir = path.dirname(dataset_path)
    cleaned_data_dir = path.join(dataset_dir, "../", "cleaned_data_handle_outliers")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(cleaned_data_dir):
        shutil.rmtree(cleaned_data_dir)
    # Create the folder
    makedirs(cleaned_data_dir, exist_ok=True)

    # Detect outliers using IQR method
    data = original_data.copy()
    outliers_IQR, cap_boundries_IQR = detect_outliers(data, DetectOutlierMethod.IQR, columns_subset)

    # Detect outliers using IQR method
    data = original_data.copy()
    outliers_zscore, cap_boundries_zscore = detect_outliers(data, DetectOutlierMethod.ZSCORE, columns_subset)

    # Drop outliers using IQR method
    data = original_data.copy()
    data_cleaned_IQR_drop = handle_outliers(data, HandleOutlierMethod.CAP_WITH_BOUNDARIES, outliers_IQR, cap_boundries_IQR)
    # Save the cleaned dataset is not empty
    if not data_cleaned_IQR_drop.empty > 0:
        data_cleaned_IQR_drop.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_IQR_drop.csv"), index=False)

    # Drop outliers using ZSCORE method
    data = original_data.copy()
    data_cleaned_zscore_drop = handle_outliers(data, HandleOutlierMethod.CAP_WITH_BOUNDARIES, outliers_zscore, cap_boundries_zscore)
    # Save the cleaned dataset is not empty
    if not data_cleaned_zscore_drop.empty > 0:
        data_cleaned_zscore_drop.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_zscore_drop.csv"), index=False)


if __name__ == "__main__":
    main()