import pandas as pd
from enum import Enum
import sys
from os import path, makedirs
import shutil
import logging


class AdjacentImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    FORWARD = 1
    BACKWARD = 2
    INTERPOLATION_LINEAR = 3
    INTERPOLATION_TIME = 4


class NumericDatatypeImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    # This enum specifies the method of imputation only for numeric columns, since for the categorical columns "MODE" is the only option
    MEAN = 1
    MEDIAN = 2
    MODE = 3


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


def handle_missing_values_drop(data: pd.DataFrame) -> pd.DataFrame:
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    logging.info(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Drop all rows containing missing values
    data = data.dropna()

    # Check dataset after dropping missing values
    logging.info(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data


def handle_missing_values_datatype_imputation(data : pd.DataFrame, numeric_datatype_imputation_method : NumericDatatypeImputationMethod) -> pd.DataFrame:
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    logging.info(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    for col in data.columns:
        # Check if the columns is numeric
        if pd.api.types.is_numeric_dtype(data[col]):
            match numeric_datatype_imputation_method:
                case NumericDatatypeImputationMethod.MEAN:
                    data[col] = data[col].fillna(data[col].mean())
                case NumericDatatypeImputationMethod.MEDIAN:
                    data[col] = data[col].fillna(data[col].median())
                case NumericDatatypeImputationMethod.MODE:
                    # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
                    data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # If it is a categorical columns
            # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
            data[col] = data[col].fillna(data[col].mode()[0])

    # Check dataset after dropping missing values
    logging.info(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data


def handle_missing_values_adjacent_value_imputation(data: pd.DataFrame, adjancent_imputation_method : AdjacentImputationMethod, time_reference_col : str = "") -> pd.DataFrame:
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    logging.info(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Fill missing values using adjacent value imputation
    match adjancent_imputation_method:
        case AdjacentImputationMethod.FORWARD:
            # Fill missing values using forward fill (Note that fillna(method='ffill') method is deprecated)
            data = data.ffill()

        case AdjacentImputationMethod.BACKWARD:
            # Fill missing values using backward fill (Note that fillna(method='bfill') method is deprecated)
            data = data.bfill()

        case AdjacentImputationMethod.INTERPOLATION_LINEAR:
            # Note that this method does not handle missing values in string columns, 
            # so it is needed to combine it with other methods to handle missing values also in string columns

            # Fill missing values using linear interpolation (Default method is linear)
            for col in data.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].interpolate(method='linear')

        case AdjacentImputationMethod.INTERPOLATION_TIME:
            # Note that this method does not handle missing values in string columns, 
            # so it is needed to combine it with other methods to handle missing values also in string columns

            # Check if time reference column is provided, as it is needed for time interpolation
            if not time_reference_col:
                logging.error("Time reference column is required for time interpolation.")
                return pd.DataFrame()
            else:
                try:
                    # Convert time reference column to datetime if contains datatime values, otherwise it will raise an error
                    data[time_reference_col] = pd.to_datetime(data[time_reference_col])
                except ValueError:
                    logging.error(f"The column '{time_reference_col}' is not in datetime format. This method needs a DataTime column to operate.")
                    return pd.DataFrame()
                
            # Change in index column to time reference column, as it is needed for time interpolation
            data = data.set_index(time_reference_col)

            # Fill missing values using time interpolation
            for col in data.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].interpolate(method='time')
            
            # Reset index to original
            data = data.reset_index()

    # Check dataset after dropping missing values
    logging.info(f"Dataset has {data.shape[0]} rows after handling missing values.")

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
                time_reference_col = ""
            case 3:
                dataset_path = sys.argv[1]
                time_reference_col = sys.argv[2]

    # Load the dataset
    original_data = load_data(dataset_path)
    # If the dataset is not valid
    if original_data.empty:
        return
    
    # Create a folder for cleaned datasets
    dataset_dir = path.dirname(dataset_path)
    output_dir = path.join(dataset_dir, "../", "output_handle_missing_values")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the folder
    makedirs(output_dir, exist_ok=True)

    # Note that I copy the original data in to a dataset for each method to assure that it works on the original dataset for experimental purpose,
    # but it is not advisable in the real-world scenarios, since it creates a big load especially for large datasets

    # Handle missing values by dropping rows with missing values
    data = original_data.copy()
    data_cleaned_drop = handle_missing_values_drop(data)
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_drop.empty:
        data_cleaned_drop.to_csv(path.join(output_dir, "dataset_cleaned_drop.csv"), index=False)

    # Handle missing values by using datatype imputation for numeric columns
    for numeric_datatype_imputation_method in list(NumericDatatypeImputationMethod):
        data = original_data.copy()
        data_cleaned_datatype_imputation = handle_missing_values_datatype_imputation(data, numeric_datatype_imputation_method)
        # Save the cleaned dataset by replacing values if the cleaned dataset is not empty
        if not data_cleaned_datatype_imputation.empty:
            data_cleaned_datatype_imputation.to_csv(path.join(output_dir, "dataset_cleaned_datatype_imputation_" + f"{numeric_datatype_imputation_method.name}.csv"), index=False)

    # Handle missing values using adjacent value imputation
    for adjacent_imputation_method in list(AdjacentImputationMethod):
        data = original_data.copy()
        data_cleaned_adjacent_value_imputation = handle_missing_values_adjacent_value_imputation(data, adjacent_imputation_method, time_reference_col)
        # Save the cleaned dataset by replacing values if the cleaned dataset is not empty
        if not data_cleaned_adjacent_value_imputation.empty:
            data_cleaned_adjacent_value_imputation.to_csv(path.join(output_dir, "dataset_cleaned_adjacent_value_imputation_" + f"{adjacent_imputation_method.name}.csv"), index=False)


if __name__ == "__main__":
    main()