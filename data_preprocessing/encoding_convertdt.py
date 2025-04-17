import pandas as pd
import sys
from os import path, makedirs
import shutil
import logging
from typing import Dict


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


def convert_datatype_auto(data : pd.DataFrame) -> pd.DataFrame:
    # Show the data types before applying any conversion
    print(f"Before automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

    # Convert data type of the numeric-like columns which has object type
    data = data.infer_objects()

    # If the data type of the columns is not numeric, it try to convert it to datatime type
    # If the column content is not datetime no changes will happen
    for col in data.columns:
        try:
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_datetime(data[col])
        except:
            pass

    # Show the data types after applying auto conversions
    print(f"After automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")

    return data    

def convert_datatype_ud(data : pd.DataFrame, convert_scenario : Dict) -> pd.DataFrame:
    # Sample conver_scenario:
    # {"column":["High School Percentage", "Test Date"],
    #  "datatype":["int", "datetime"],
    #  "format":["", "%m/%d/%Y"] }

    # Show the data types before applying any conversion
    print(f"Before automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")
    
    # Check all the provided column names to be in the dataset
    if not all(col in data.columns for col in convert_scenario["column"]):
        logging.error("At least one of the columns provides in the scenario is not valid!")
        return data
    
    # Check all the provided datatypes to be valid
    if not all(dt in ["float", "int", "datetime"] for dt in convert_scenario["datatype"]):
        logging.error("At least one of the type provides in the scenario is not valid! The only acceptable data types are: {float, int, datetime}")
        return data
    
    # create a list of tuples based on the convert_scenario dict
    # zip command creates tuple of the elements of a row
    # conversion to a list is necessary since zipped output can be consumed only once, but we need it more
    convert_scenario_zipped = list(zip(convert_scenario["column"], convert_scenario["datatype"], convert_scenario["format"]))

    # Check if there is a non datetime column which has provided format (we accept format only for datetime conversion)
    if any(ft != "" and dt != "datetime" for _ , dt, ft in convert_scenario_zipped):
        logging.error("Only datetime conversion accepts format (ISO 8601)")
        return data
    
    # Check all the datetime conversions have format
    if any(ft == "" and dt == "datetime" for _ , dt, ft in convert_scenario_zipped):
        logging.error("Datetime conversion needs format (ISO 8601)")
        return data
    
    for col, dt, ft in convert_scenario_zipped:
        try:
            match dt:
                case "int":
                    # If the current type is not int, the conversion will apply
                    if not pd.api.types.is_integer_dtype(data[col]):
                        data[col] = data[col].astype("int")
                case "float":
                    # If the current type is not int, the conversion will apply
                    if not pd.api.types.is_float_dtype(data[col]):
                        data[col] = data[col].astype("float")
                case "datetime":
                    # If the current type is not numeric and datetime, the conversion will apply
                    if not pd.api.types.is_datetime64_any_dtype(data[col]) and not pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_datetime(data[col], format=ft)
        except Exception as e:
            logging.error(f"Conversion failed for column '{col}' with error: {e}")
            return data
    
    # Show the data types after applying auto conversions
    print(f"After automatic datatype conversion, the datatype are as follows:\n{data.dtypes}")
    print(data)
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
                duplicate_columns_subset = None
                ratio_range = None
            case 3:
                dataset_path = sys.argv[1]
                # This parameter should pass to the program in comma seperated format e.g., "First Name,Last Name" (Obviously column names are case-sebsitive)
                if sys.argv[2] == "None":
                    duplicate_columns_subset = None
                else:
                    duplicate_columns_subset = sys.argv[2].split(",")
                ratio_range = None
            case 4:
                dataset_path = sys.argv[1]
                # This parameter should pass to the program in comma seperated format e.g., "First Name,Last Name" (Obviously column names are case-sebsitive)
                if sys.argv[2] == "None":
                    duplicate_columns_subset = None
                else:
                    duplicate_columns_subset = sys.argv[2].split(",")
                # Ratio range should be passed as a tuple e.g., 80,90
                ratio_range = tuple([int(item) for item in sys.argv[3].split(",")])


    # Load the dataset
    original_data = load_data(dataset_path)
    # If the dataset is not valid
    if original_data.empty:
        return
    
    # Create a folder for cleaned datasets
    dataset_dir = path.dirname(dataset_path)
    output_dir = path.join(dataset_dir, "../", "output_handle_duplicate_values")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the folder
    makedirs(output_dir, exist_ok=True)

    data = original_data.copy()
    # convert_datatype_auto(data)

    convert_datatype_ud(data, {"column":["High School Percentage", "Admission Test Score", "Test Date"], "datatype":["int", "float", "datetime"], "format":["","", "%m/%d/%Y"]})


if __name__ == "__main__":
    main()