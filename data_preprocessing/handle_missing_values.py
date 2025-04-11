import pandas as pd
from enum import Enum
import sys
from os import path, makedirs


class AdjacentImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    Forward = 1
    Backward = 2
    Interpolation_Linear = 3
    Interpolation_Time = 4


def load_data(file_path : str) -> pd.DataFrame:
    # Open csv file and load it into a dataframe
    try:
        data = pd.read_csv(file_path)
    except:
        print("The path is invalid!")
        return pd.DataFrame()

    # Return the first 5 rows of the dataset
    print(data.head())

    return data


def handle_missing_values_drop(data: pd.DataFrame) -> pd.DataFrame:
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    print(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Drop all rows containing missing values
    data.dropna(inplace=True)

    # Check dataset after dropping missing values
    print(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data


def handle_missing_values_adjacent_value_imputation(data: pd.DataFrame, adjancent_imputation_method : AdjacentImputationMethod, time_reference_col : str = "") -> pd.DataFrame:
    # Check for missing values
    # It is also possible to use isnull() instead of isna()
    print(f"Dataset has {data.shape[0]} rows before handling missing values.\nMissing values are:\n{data.isna().sum()}")

    # Fill missing values using adjacent value imputation
    match adjancent_imputation_method:
        case AdjacentImputationMethod.Forward:
            # Fill missing values using forward fill (Note that fillna(method='ffill') method is deprecated)
            data.bfill(inplace=True)

        case AdjacentImputationMethod.Backward:
            # Fill missing values using backward fill (Note that fillna(method='bfill') method is deprecated)
            data.ffill(inplace=True)

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
                print("Time reference column is required for time interpolation.")
                return data
            else:
                try:
                    # Convert time reference column to datetime if contains datatime values, otherwise it will raise an error
                    data[time_reference_col] = pd.to_datetime(data[time_reference_col])
                except ValueError:
                    print(f"The column '{time_reference_col}' is not in datetime format. This method needs a DataTime column to operate.")
                    return data
                
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
    print(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data


def main():
    # Check the argument passed to the program
    # Note that the first argument is always the name of the program
    if len(sys.argv) < 2:
        print("This program needs at least one parameter as the dataset path!")
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
    makedirs(cleaned_data_dir,exist_ok=True)

    # Note that I copy the original data in to a dataset for each method to assure that it works on the original dataset for experimental purpose
    # but it is not advisable in the real-world scenarios, since it creates a big load especially for large datasets

    # Handle missing values by dropping rows with missing values
    data = original_data.copy()
    data_cleaned_drop = handle_missing_values_drop(data)
    # Save the cleaned dataset by dropping rows
    data_cleaned_drop.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_drop.csv"), index=False)

 
    # Handle missing values using adjacent value imputation Forward
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_forward = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Forward)
    # Save the cleaned dataset by forward imputation 
    data_cleaned_adjacent_value_imputation_forward.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_forward.csv"), index=False)

    # Handle missing values using adjacent value imputation Backward
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_backward = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Backward)
    # Save the cleaned dataset by backward imputation
    data_cleaned_adjacent_value_imputation_backward.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_backweard.csv"), index=False)
    
    # Handle missing values using adjacent value imputation Interpolation_Linear
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_interpolation_linear = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Interpolation_Linear)
    # Save the cleaned dataset by linear interpolation
    data_cleaned_adjacent_value_imputation_interpolation_linear.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_interpolation_linear.csv"), index=False)
    
    # Handle missing values using adjacent value imputation Interpolation_Time
    data = original_data.copy()
    data_cleaned_adjacent_value_imputation_interpolation_time = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Interpolation_Time, time_reference_col)
    # Save the cleaned dataset by time interpolation
    data_cleaned_adjacent_value_imputation_interpolation_time.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_adjacent_value_imputation_interpolation_time.csv"), index=False)
    
    
if __name__ == "__main__":
    main()