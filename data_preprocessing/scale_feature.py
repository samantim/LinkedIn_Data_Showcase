import pandas as pd
import sys
from os import path, makedirs
import shutil
import logging
from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from enum import Enum


class ScalingMethod(Enum):
    MINMAX_SCALING = 1
    ZSCORE_STANDARDIZATION = 2
    ROBUST_SCALING = 3


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


def scale_feature(data : pd.DataFrame, scale_scenario : Dict, apply_l2normalization : bool = False) -> pd.DataFrame:
    # Sample scale_scenario:
    # {"column":["High School Percentage", "Age"],
    #  "scaling_method":["ROBUST_SCALING", "ZSCORE_STANDARDIZATION"]}

    if len(scale_scenario["column"]) != len(scale_scenario["scaling_method"]):
        logging.error("Number of columns and scaling methods do not match!")
        return data

    # Check if column_subset is valid
    observing_columns = get_observing_columns(data, scale_scenario["column"])
    if len(observing_columns) == 0: return data

    scale_scenario["scaling_method"] = [sm.strip() for sm in scale_scenario["scaling_method"]]
    # Check all the provided scaling_method to be valid
    if not all(sm in [c.name for c in list(ScalingMethod)] for sm in scale_scenario["scaling_method"]):
        logging.error("At least one of the scaling methods provided in the scenario is not valid! The only acceptable data types are: {MINMAX_SCALING, ZSCORE_STANDARDIZATION, ROBUST_SCALING}")
        return data

    # Create a list of tuples (column, scaling_method)
    scale_scenario_zipped = list(zip(observing_columns,scale_scenario["scaling_method"]))

    # For each column in the list, we apply the proper scaling method
    # Then update the date[column]
    for column, scaling_method in scale_scenario_zipped:
        match scaling_method:
            case ScalingMethod.MINMAX_SCALING.name:
                minmax_scaler = MinMaxScaler()
                data[column] = minmax_scaler.fit_transform(data[[column]])
            case ScalingMethod.ZSCORE_STANDARDIZATION.name:
                zscore_standardization = StandardScaler()
                data[column] = zscore_standardization.fit_transform(data[[column]])
            case ScalingMethod.ROBUST_SCALING.name:
                robust_scaler = RobustScaler()
                data[column] = robust_scaler.fit_transform(data[[column]])

    # If apply_l2normalization then we apply l2 normalization on all numeric columns of the dataset
    # Since this type of normalization only makes sense if it applies on all numeric columns
    if apply_l2normalization:
        l2_normalizer = Normalizer()
        # Extract all numeric columns
        numeric_columns = data.select_dtypes("number")
        data_transformed = l2_normalizer.fit_transform(numeric_columns)
        col_number = 0
        for col in numeric_columns.columns: 
            # Update dataset based on all numeric columns which are normalized
            data[col] = data_transformed[:,col_number]
            col_number += 1 

    return data

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
    output_dir = path.join(dataset_dir, "../", "output_scale_feature")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the folder
    makedirs(output_dir, exist_ok=True)

    # Scale and normalize the dataset
    data = original_data.copy()
    data_scaled = scale_feature(data, {"column":columns_subset, "scaling_method":columns_scaling_method}, apply_l2normalization)
    # Save the converted dataset if the it is not empty
    if not data_scaled.empty:
        data_scaled.to_csv(path.join(output_dir, "dataset_scaled.csv"), index=False)
    

if __name__ == "__main__":
    main()