import pandas as pd
from enum import Enum


class AdjacentImputationMethod(Enum):
    # Enum classes make the code cleaner and avoid using invalid inputs
    Forward = 1
    Backward = 2
    Interpolation_Linear = 3
    Interpolation_Time = 4

def load_data(file_path : str) -> pd.DataFrame:
    # Open csv file and load it into a dataframe
    data = pd.read_csv(file_path)

    # Return the first 5 rows of the dataset
    print(data.head())

    return data


# take hadling method dropna, fillna, fillba, mode, mean, median
# general approach

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
            # Fill missing values using forward fill (Note that fillna() method is deprecated)
            data.bfill(inplace=True)
        case AdjacentImputationMethod.Backward:
            # Fill missing values using backward fill (Note that fillna() method is deprecated)
            data.ffill(inplace=True)
        case AdjacentImputationMethod.Interpolation_Linear:
            # Change columns with numbers looking like strings to numbers which is needed for interpolation as it only works with numbers
            # Note that this method does not handle missing values in string columns
            data = change_numberlooking_columns_to_number(data)
            # Fill missing values using linear interpolation (Default method is linear)
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].interpolate(method='linear', inplace=True)
        case AdjacentImputationMethod.Interpolation_Time:
            if not time_reference_col:
                raise ValueError("Time reference column is required for time interpolation.")
            # Change in index column to time reference column as it is needed for time interpolation
            data.set_index(time_reference_col, inplace=True)
            # Fill missing values using time interpolation
            data.interpolate(method='time', inplace=True)

    # Check dataset after dropping missing values
    print(f"Dataset has {data.shape[0]} rows after handling missing values.")

    return data

def change_numberlooking_columns_to_number(data: pd.DataFrame) -> pd.DataFrame:
    # Change columns with numbers looking like strings to numbers
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col])
            except ValueError:
                pass

    return data

def main():
    # Load the dataset
    data = load_data('dataset.csv')

    # print(data.dtypes)

    # Handle missing values by dropping rows with missing values
    # data = handle_missing_values_drop(data)

    # Handle missing values using adjacent value imputation
    data = handle_missing_values_adjacent_value_imputation(data, AdjacentImputationMethod.Interpolation_Linear)

    print(data.dtypes)
    
    # Save the cleaned dataset to a new CSV file
    data.to_csv("dataset_cleaned.csv", index=False)

if __name__ == "__main__":
    main()