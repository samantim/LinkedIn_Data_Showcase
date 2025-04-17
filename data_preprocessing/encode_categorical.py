import pandas as pd
import sys
from os import path, makedirs
import shutil
import logging
from typing import List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
from enum import Enum
from math import log2, ceil


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


def encode_categorical(data : pd.DataFrame, categorical_encoding_method : CategoricalEncodingMethod, columns_subset : List = None) -> pd.DataFrame:
    # Check if column_subset is valid
    observing_columns = get_observing_columns(data, columns_subset)
    if len(observing_columns) == 0: return data

    match categorical_encoding_method:
        # Encode the observing columns using label encoder
        case CategoricalEncodingMethod.LABEL_ENCODING:
            label_encoder = LabelEncoder()
            for col in observing_columns:
                # The names of the new columns are defined and concat them to end of original columns
                data["_".join([col, "encoded"])] = label_encoder.fit_transform(data[col])

        case CategoricalEncodingMethod.ONEHOT_ENCODING:
            # Encode observing columns using one-hot encoder
            onehot_encoder = OneHotEncoder(sparse_output=False)
            for col in observing_columns:
                # Create the encoded contents
                encoded_columns = onehot_encoder.fit_transform(data[[col]])
                # The column names also created by the model
                encoded_columns_name = onehot_encoder.get_feature_names_out([col])
                # Create a dataframe with contents and the column names
                encoded_df = pd.DataFrame(data = encoded_columns, columns=encoded_columns_name)
                # Concat the new datafram to end of original columns
                data = pd.concat([data, encoded_df], axis=1)

        case CategoricalEncodingMethod.HASHING:
            for col in observing_columns:
                # Find the log2 of the number of categories (number of unique values in the columns)
                # It is the optimal number of components to reduce the collisions while having good performance
                unique_len = len(data[col].unique())
                # Throw a warning for the less number of unique values
                if unique_len < 10: logging.warning(f"Hashing for category number less than 10 is not reasonable (column='{col}', category number={unique_len}), and the results would not be promising!")
                n_components=ceil(log2(unique_len))
                # Encode the observing columns using hashing encoder
                hashing = ce.HashingEncoder(n_components = n_components, return_df=True)
                # Create the encoded contents
                encoded_columns = hashing.fit_transform(data[[col]])
                # Enhance the names for more clarification
                encoded_columns.columns = ["_".join([col,col_gen_name]) for col_gen_name in encoded_columns.columns]
                # Concat the new datafram to end of original columns
                data = pd.concat([data, encoded_columns], axis=1)

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
    output_dir = path.join(dataset_dir, "../", "output_encode_categorical")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the folder
    makedirs(output_dir, exist_ok=True)

    # Encode categorical columns by label encoding
    data = original_data.copy()
    data_converted = encode_categorical(data, CategoricalEncodingMethod.LABEL_ENCODING, columns_subset=columns_subset)
    # Save the converted dataset if the it is not empty
    if not data_converted.empty:
        data_converted.to_csv(path.join(output_dir, "dataset_label_encoding.csv"), index=False)

    # Encode categorical columns by onehot encoding
    data = original_data.copy()
    data_converted = encode_categorical(data, CategoricalEncodingMethod.ONEHOT_ENCODING, columns_subset=columns_subset)
    # Save the converted dataset if the it is not empty
    if not data_converted.empty:
        data_converted.to_csv(path.join(output_dir, "dataset_onehot_encoding.csv"), index=False)

    # Encode categorical columns by hashing
    data = original_data.copy()
    data_converted = encode_categorical(data, CategoricalEncodingMethod.HASHING, columns_subset=columns_subset)
    # Save the converted dataset if the it is not empty
    if not data_converted.empty:
        data_converted.to_csv(path.join(output_dir, "dataset_hashing.csv"), index=False)


if __name__ == "__main__":
    main()