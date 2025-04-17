import pandas as pd
import numpy as np
from enum import Enum
import sys
from os import path, makedirs
from typing import List, Dict, Tuple
import shutil
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

class DetectOutlierMethod(Enum):
    IQR = 1
    ZSCORE = 2
    ISOLATION_FOREST = 3
    LOCAL_OUTLIER_FACTOR = 4


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


def detect_outliers(data : pd.DataFrame, detect_outlier_method : DetectOutlierMethod, columns_subset : List = None, contamination_rate : float | str = "auto" , n_neighbors : int = 20, per_column_detection : bool = True) -> Tuple:
    # Parameter contamination_rate is used for training ISOLATION FOREST LOCAL OUTLIER FACTOR methods to set the boundries for outliers
    # Parameter n_neighbors is used for training OUTLIER FACTOR methods to set the number of observing neighbors
    # Parameter per_column_detection is used for training ISOLATION FOREST LOCAL OUTLIER FACTOR methods, as their main usage in analyzing multivariate data rather than univariates
    # but in case of comparison, I include both per-column and all-columns approaches for these methods

    # Check if column_subset is valid
    observing_columns = get_observing_columns(data, columns_subset)
    if len(observing_columns) == 0: return dict(), dict()

    # Validating contamination_rate
    if contamination_rate != "auto":
        try:
            contamination_rate = float(contamination_rate)
            if not (0.0 < contamination_rate <= 0.5):
                raise ValueError
        except (TypeError, ValueError):
            logging.error("The 'contamination' parameter of IsolationForest must be a str among {'auto'} or a float in the range (0.0, 0.5]")
            return dict(), dict()

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
            if per_column_detection:
                for col in observing_columns:
                    # Create an object of the IsolationForest class, then train it and get the predictions based on the contamination_rate
                    isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                    # Be careful about the input formation. e.g., fit_predict accepts dataframe not a series. So, should give data[[col]] not data[col]
                    predictions = isolation_forest.fit_predict(data[[col]])
                    # if the prediction == -1 means it is an outlier
                    outliers[col] = data[predictions == -1].index.to_list()
                    # Isolation forest method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())
            else:
                # Create an object of the IsolationForest class, then train it and get the predictions based on the contamination_rate
                isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                # Train the model based on all columns under observation
                predictions = isolation_forest.fit_predict(data[observing_columns])
                # if the prediction == -1 means it is an outlier
                outliers_indexes = data[predictions == -1].index.to_list()
                for col in observing_columns:
                    outliers[col] = outliers_indexes
                    # Isolation forest method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())

        case DetectOutlierMethod.LOCAL_OUTLIER_FACTOR:
            if per_column_detection:
                for col in observing_columns:
                    # Create an object of the LocalOutlierFactor class, then train it and get the predictions based on the contamination_rate and n_neighbors
                    local_outlier_factor = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=n_neighbors)
                    # Be careful about the input formation. e.g., fit_predict accepts dataframe not a series. So, should give data[[col]] not data[col]
                    predictions = local_outlier_factor.fit_predict(data[[col]])
                    # if the prediction == -1 means it is an outlier
                    outliers[col] = data[predictions == -1].index.to_list()
                    # Local outlier factor method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())
            else:
                # Create an object of the LocalOutlierFactor class, then train it and get the predictions based on the contamination_rate and n_neighbors
                local_outlier_factor = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=n_neighbors)
                # Train the model based on all columns under observation
                predictions = local_outlier_factor.fit_predict(data[observing_columns])
                # if the prediction == -1 means it is an outlier
                outliers_indexes = data[predictions == -1].index.to_list()
                for col in observing_columns:
                    outliers[col] = outliers_indexes
                    # Local outlier factor method, rather than IQR and Z-Score, does not have native boundries. So, we use the min and max value of inliers for that
                    inliers = data.loc[~data.index.isin(outliers[col]), col]
                    boundries[col] = (inliers.min(), inliers.max())

    # Output of the function is a Tuple consists of oulier indexes dict and boundries on inliers dict
    return outliers, boundries


def handle_outliers(data : pd.DataFrame, handle_outlier_method : HandleOutlierMethod, outliers : Dict = {}, boundries : Dict = {}) -> pd.DataFrame :
    # If the outlier dict is empty, the output is the original data
    if len(outliers) == 0: return data

    # First unpack the values of the outlier dict and then union all of them in a set (to eliminate duplicate indexes)
    all_drop_indexes = set().union(*outliers.values())
    
    # Check dataset to know how many duplicate values exist
    # Find duplicate values
    logging.info(f"Dataset has {data.shape[0]} rows before handling outliers values.\nTop 10 of rows containing outliers are (Totally {len(all_drop_indexes)} rows):\n{data.iloc[list(all_drop_indexes)].head(10)}")

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
                # If the boundries are float, the column type should be converted to float (implicit casting is deprecated)
                # "isinstance" is safer than "type", since it also include numpy types
                if isinstance(lower, float) or isinstance(upper, float):
                    data[col] = data[col].astype(float)
                data.loc[outliers[col], col] = data.loc[outliers[col], col].clip(lower, upper)

    # Check dataset rows after removing duplicate rows
    logging.info(f"Dataset has {data.shape[0]} rows after handling outliers.")

    return data


def visualize_outliers(original_data : pd.DataFrame, cleaned_data : pd.DataFrame, output_dir : str, detect_outlier_method : DetectOutlierMethod, handle_outlier_method : HandleOutlierMethod, columns_subset : List = None):
    # Check if column_subset is valid
    observing_columns = get_observing_columns(original_data, columns_subset)
    if len(observing_columns) == 0: return

    # Make the visualization directory path
    visualization_dir = path.join(output_dir, "visualizations")
    # Create the folder
    makedirs(visualization_dir, exist_ok=True)

    # Set the resolution and quality
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # Setup the layout to fit in the figure
    plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)   
    
    for col in observing_columns:
        plt.subplot(2,2,1)
        sns.boxplot(original_data[col])

        plt.subplot(2,2,2)
        sns.histplot(original_data[col], kde=True)

        plt.subplot(2,2,3)
        sns.boxplot(cleaned_data[col])

        plt.subplot(2,2,4)
        sns.histplot(cleaned_data[col], kde=True)

        # Save the file with proper dpi
        file_name = path.join(visualization_dir, "_".join([detect_outlier_method.name, handle_outlier_method.name, col]) + ".png")
        plt.savefig(fname=file_name, format="png", dpi=fig.dpi)

        plt.clf()
        
    plt.close()
    
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
    output_dir = path.join(dataset_dir, "../", "output_handle_outliers")
    # Remove the directory if exists because some of the files may not need to create based on the program arguments
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create the folder
    makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # Detect outliers by all methods including IQR, ZSCORE, ISOLATION_FOREST, LOCAL_OUTLIER_FACTOR
    # -------------------------------

    outliers = dict()
    cap_boundries = dict()

    # Detect outliers
    for detect_outlier_method in list(DetectOutlierMethod):
        outliers[detect_outlier_method], cap_boundries[detect_outlier_method] = detect_outliers(original_data, detect_outlier_method, columns_subset,"auto",20,False)

    # -------------------------------
    # Handle outliers by DROP, REPLACE_WITH_MEDIAN, CAP_WITH_BOUNDARIES methods based on outliers detected via IQR, ZSCORE, ISOLATION_FOREST, LOCAL_OUTLIER_FACTOR methods + Visualizations
    # -------------------------------

    # Drop outliers
    for detect_outlier_method in list(DetectOutlierMethod):
        for handle_outlier_method in list(HandleOutlierMethod):
            data = original_data.copy()
            data_cleaned = handle_outliers(data, handle_outlier_method, outliers[detect_outlier_method], cap_boundries[detect_outlier_method])        
            # Save the cleaned dataset is not empty
            if not data_cleaned.empty:
                data_cleaned.to_csv(path.join(output_dir, "_".join(["dataset_cleaned", detect_outlier_method.name, handle_outlier_method.name]) + ".csv"), index=False)
            visualize_outliers(original_data, data_cleaned, output_dir, detect_outlier_method, handle_outlier_method, columns_subset)


if __name__ == "__main__":
    main()