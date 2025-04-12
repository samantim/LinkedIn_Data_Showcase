import pandas as pd
import numpy as np
import sys
from os import path, makedirs
import shutil
import logging
from typing import List, Tuple
from rapidfuzz import fuzz
from itertools import combinations

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
    # here we use keep='first' (default), since we need to keep the first one from each group of duplicates
    data = data.drop_duplicates(subset=subset)

    # Check dataset rows after removing duplicate rows
    logging.info(f"Dataset has {data.shape[0]} rows before handling duplicate values.")

    return data


def handle_duplicate_values_fuzzy(data : pd.DataFrame, subset : List = None, ratio_range : Tuple = None) -> pd.DataFrame:
    # Note that if ratio_range(100,100) is given to the function, the results are identical to handle_duplicate_values_drop() function

    # If ratio_range is not passed to the function it will be considered as (90,100)
    if ratio_range is None:
        ratio_range = (90,100)

    # If a subset of columns is given, we only consider these columns in similarity comparisons
    # If it is not assigned, we use all columns (It is better to give all categorical columns to the function, as the fuzz method is basically for string matching)
    comparison_columns = subset if subset else data.columns


    # This is a list containing sets of indexes. each set is for a group of duplicates.
    data_duplicated_sets = []
    # This is a list containing the similarity ratios of each column of under-comparison rows
    column_similarity_ratios = []
    # Iteration is on every unique non-ordered combination of the row indexes
    for i, j in combinations(data.index, 2):
        # For each comparison column, similarity ratio is calculated
        for col in comparison_columns:
            # The result is stored in ratios list
            column_similarity_ratios.append(fuzz.ratio(str(data.loc[i, col]).lower().strip(), str(data.loc[j, col]).lower().strip()))
        
        # Average of similarity ratios of all column is caculated.
        rows_similarity_avg_ratio = sum(column_similarity_ratios)/len(column_similarity_ratios)
        # If the result is in range, those rows will be considered as duplicates
        if ratio_range[0] <= rows_similarity_avg_ratio <= ratio_range[1]:
            # If it is the first group of duplicates, we add them without question
            if len(data_duplicated_sets) == 0:
                # Create a new set and add it to the list
                new_duplicated_set = set([i, j])
                data_duplicated_sets.append(new_duplicated_set)
            else:
                # It shows if we need to create a new set or add the indexes to the existing one
                new_set = True
                # We search if each item of our newly found pair is in the existing sets
                for d_set in data_duplicated_sets:
                    # If they exist, simply add both of them to the set
                    if i in d_set or j in d_set:
                        d_set.add(i)
                        d_set.add(j)
                        # No need to create a new set
                        new_set = False
                        break
                if new_set:
                    # If they don't exist, we need to create a new set of duplicates and add it to list
                    new_duplicated_set = set([i, j])
                    data_duplicated_sets.append(new_duplicated_set)
        
        # For each combination of rows, we need fresh ration list
        column_similarity_ratios.clear()

    # Based on the logic in handle_duplicate_values_drop() function, we show all the duplicated rows to the user (keep = False),
    # but should not eliminate the first duplicated row (keep='first')
    # So, need to have a set of indexes
    data_duplicated_set_show = set()
    data_duplicated_set_drop = set()
    for d_set in data_duplicated_sets:
        # Union all the sets of indexes (each group of duplicates) into one set, for further operations
        data_duplicated_set_show = data_duplicated_set_show.union(d_set)
        # Remove the first element of each group and union the others into one set
        first_duplicate = sorted(list(d_set))[0]
        d_set.remove(first_duplicate)
        data_duplicated_set_drop = data_duplicated_set_drop.union(d_set)

    # Convert sets to sorted lists
    data_duplicated_index_show = sorted(list(data_duplicated_set_show))
    data_duplicated_index_drop = sorted(list(data_duplicated_set_drop))

    # Check dataset to know how many duplicate values exist
    # Find duplicate values
    logging.info(f"Dataset has {data.shape[0]} rows before handling duplicate values.\nTop 10 of duplicate values are (Totally {len(data_duplicated_index_show)} rows - including all duplicates, but from each group first one will remain and others will be removed):\n{data.iloc[data_duplicated_index_show]}")

    # Remove duplicate values
    data = data.drop(data_duplicated_index_drop)

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
    cleaned_data_dir = path.join(dataset_dir, "../", "cleaned_data_handle_duplicate_values")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(cleaned_data_dir):
        shutil.rmtree(cleaned_data_dir)
    # Create the folder
    makedirs(cleaned_data_dir, exist_ok=True)

    # Handle duplicate values using drop method
    data = original_data.copy()
    data_cleaned_drop = handle_duplicate_values_drop(data=data, subset=duplicate_columns_subset)
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_drop.empty:
        data_cleaned_drop.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_drop.csv"), index=False)

    # Handle duplicate values using fuzzy method
    data = original_data.copy()
    data_cleaned_fuzzy = handle_duplicate_values_fuzzy(data=data, subset=duplicate_columns_subset, ratio_range = ratio_range)  
    # Save the cleaned dataset by dropping rows if the cleaned dataset is not empty
    if not data_cleaned_fuzzy.empty:
        data_cleaned_fuzzy.to_csv(path.join(cleaned_data_dir, "dataset_cleaned_fuzzy.csv"), index=False)


if __name__ == "__main__":
    main()