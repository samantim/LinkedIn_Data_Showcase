import pandas as pd
from enum import Enum
import sys
from os import path, makedirs
import shutil
import logging


class AdjacentImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    Forward = 1
    Backward = 2
    Interpolation_Linear = 3
    Interpolation_Time = 4

class NumericDatatypeImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    # This enum specifies the method of imputation only for numeric columns, since for the categorical columns "MODE" is the only option
    MEAN = 1
    MEDIAN = 2
    MODE = 3


def load_data(file_path : str) -> pd.DataFrame:
    # Open csv file and load it into a dataframe
    try:
        data = pd.read_csv(file_path)
    except:
        logging.error("The path is invalid!")
        return pd.DataFrame()

    # Return the first 5 rows of the dataset
    logging.info(data.head())

    return data


def handle_missing_values_drop(data: pd.DataFrame) -> pd.DataFrame:
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    logging.info(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Drop all rows containing missing values
    data.dropna(inplace=True)

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
                    data[col].fillna(data[col].mean(), inplace=True)
                case NumericDatatypeImputationMethod.MEDIAN:
                    data[col].fillna(data[col].median(), inplace=True)
                case NumericDatatypeImputationMethod.MODE:
                    # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
                    data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            # If it is a categorical columns
            # Note that in this case, mode method returns a data serie which contains all modes, so we need to take the first one
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Check dataset after dropping missing values
    logging.info(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data


def handle_missing_values_adjacent_value_imputation(data: pd.DataFrame, adjancent_imputation_method : AdjacentImputationMethod, time_reference_col : str = "") -> pd.DataFrame:
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    logging.info(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Fill missing values using adjacent value imputation
    match adjancent_imputation_method:
        case AdjacentImputationMethod.Forward:
            # Fill missing values using forward fill (Note that fillna(method='ffill') method is deprecated)
            data.ffill(inplace=True)

        case AdjacentImputationMethod.Backward:
            # Fill missing values using backward fill (Note that fillna(method='bfill') method is deprecated)
            data.bfill(inplace=True)

        case AdjacentImputationMethod.Interpolation_Linear:
            # Note that this method does not handle missing values in string columns, 
            # so it is needed to combine it with other methods to handle missing values also in string columns

            # Fill missing values using linear interpolation (Default method is linear)
            for col in data.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].interpolate(method='linear')

        case AdjacentImputationMethod.Interpolation_Time:
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
            data.set_index(time_reference_col, inplace=True)

            # Fill missing values using time interpolation
            for col in data.columns:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].interpolate(method='time')
            
            # Reset index to original
            data.reset_index(inplace=True)

    # Check dataset after dropping missing values
    logging.info(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data

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
    cleaned_data_dir = path.join(dataset_dir, "cleaned_data")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    shutil.rmtree(cleaned_data_dir)
    # Create the folder
    makedirs(cleaned_data_dir, exist_ok=True)

    # Note that I copy the original data in to a dataset for each method to assure that it works on the original dataset for experimental purpose,
    # but it is not advisable in the real-world scenarios, since it creates a big load especially for large datasets

    # Handle missing values by dropping rows with missing values
    data = original_data.copy()
    data_cleaned_drop = handle_missing_values_drop(data)
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_drop.empty:
        data_cleaned_drop.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_drop.csv"), index=False)

    # Handle missing values by using datatype imputation Mean for numeric columns
    data = original_data.copy()
    data_cleaned_datatype_imputation_mean = handle_missing_values_datatype_imputation(data, NumericDatatypeImputationMethod.MEAN)
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_datatype_imputation_mean.empty:
        data_cleaned_datatype_imputation_mean.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_datatype_imputation_mean.csv"), index=False)
 
    # Handle missing values by using datatype imputation Median for numeric columns
    data = original_data.copy()
    data_cleaned_datatype_imputation_median = handle_missing_values_datatype_imputation(data, NumericDatatypeImputationMethod.MEDIAN)
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_datatype_imputation_median.empty:
        data_cleaned_datatype_imputation_median.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_datatype_imputation_median.csv"), index=False)

    # Handle missing values by using datatype imputation Mode for numeric columns
    data = original_data.copy()
    data_cleaned_datatype_imputation_mode = handle_missing_values_datatype_imputation(data, NumericDatatypeImputationMethod.MODE)
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_datatype_imputation_mode.empty:
        data_cleaned_datatype_imputation_mode.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_datatype_imputation_mode.csv"), index=False)

    # Handle missing values using adjacent value imputation Forward
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_forward = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Forward)
    # Save the cleaned dataset by forward imputation if the cleaned dataset is not empty
    if not data_cleaned_adjacent_value_imputation_forward.empty:
        data_cleaned_adjacent_value_imputation_forward.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_forward.csv"), index=False)

    # Handle missing values using adjacent value imputation Backward
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_backward = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Backward)
    # Save the cleaned dataset by backward imputation if the cleaned dataset is not empty
    if not data_cleaned_adjacent_value_imputation_backward.empty:
        data_cleaned_adjacent_value_imputation_backward.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_backward.csv"), index=False)
    
    # Handle missing values using adjacent value imputation Interpolation_Linear
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_interpolation_linear = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Interpolation_Linear)
    # Save the cleaned dataset by linear interpolation if the cleaned dataset is not empty
    if not data_cleaned_adjacent_value_imputation_interpolation_linear.empty:
        data_cleaned_adjacent_value_imputation_interpolation_linear.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_interpolation_linear.csv"), index=False)
    
    # Handle missing values using adjacent value imputation Interpolation_Time
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_interpolation_time = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Interpolation_Time, time_reference_col)
    # Save the cleaned dataset by time interpolation if the cleaned dataset is not empty
    if not data_cleaned_adjacent_value_imputation_interpolation_time.empty:
        data_cleaned_adjacent_value_imputation_interpolation_time.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_interpolation_time.csv"), index=False)
    

if __name__ == "__main__":
    main()