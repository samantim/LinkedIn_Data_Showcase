import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from handle_missing_values import (
    handle_missing_values_drop,
    handle_missing_values_datatype_imputation,
    handle_missing_values_adjacent_value_imputation,
    NumericDatatypeImputationMethod,
    AdjacentImputationMethod,
)


@pytest.fixture
def sample_df1():
    return pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": [None, "x", "y", None],
        "C": pd.date_range("2023-01-01", periods=4)
    })


@pytest.fixture
def sample_df2():
    # This will create a DataFrame for testing purposes
    data = {
        "A": [1, 2, None, 4],
        "B": [None, 2, 3, 4],
        "C": ["a", "a", "c", None]
    }
    return pd.DataFrame(data)


# -------------------------------
# Test handle_missing_values_drop
# -------------------------------

def test_handle_missing_values_drop(sample_df1):
    result = handle_missing_values_drop(sample_df1.copy())
    assert result.isna().sum().sum() == 0
    # Only the second row has no missing values
    assert len(result) == 1  

# -------------------------------
# Test handle_missing_values_datatype_imputation (MEAN)
# -------------------------------

def test_handle_missing_values_datatype_imputation_mean1(sample_df1):
    df = sample_df1.copy()
    df.loc[0, "A"] = None
    result = handle_missing_values_datatype_imputation(df, NumericDatatypeImputationMethod.MEAN)
    assert result["A"].isna().sum() == 0
    # Mean([2, 4]) == 3
    assert result.loc[0, "A"] == 3


def test_handle_missing_values_datatype_imputation_mean2(sample_df2):
    data = handle_missing_values_datatype_imputation(sample_df2, NumericDatatypeImputationMethod.MEAN)

    # Mean of column "A" -> (1 + 2 + 4) / 3 = 2.3333...
    assert data["A"][2] == pytest.approx(2.333333, rel=1e-5)  
    # Mean of column "B" -> (2 + 3 + 4) / 3 = 3.0
    assert data["B"][0] == pytest.approx(3.0, rel=1e-5)  
    # "C" is a categorical column and always is changed by Mode method, no matter what the input enum is
    assert data["C"][3] == "a" 

# -------------------------------
# Test handle_missing_values_datatype_imputation (MEDIAN)
# -------------------------------

def test_handle_missing_values_datatype_imputation_median1(sample_df1):
    df = sample_df1.copy()
    result = handle_missing_values_datatype_imputation(df, NumericDatatypeImputationMethod.MEDIAN)
    assert result["A"].isna().sum() == 0
    # Median([1,2,4]) == 2
    assert result.loc[2, "A"] == 2


def test_handle_missing_values_datatype_imputation_median2(sample_df2):
    data = handle_missing_values_datatype_imputation(sample_df2, NumericDatatypeImputationMethod.MEDIAN)

    # Median of column "A" -> (2 is the median)
    assert data["A"][2] == 2.0 
    # Median of column "B" -> (3 is the median)
    assert data["B"][0] == 3.0
    # "C" is a categorical column and always is changed by Mode method, no matter what the input enum is
    assert data["C"][3] == "a"  

# -------------------------------
# Test handle_missing_values_datatype_imputation (MODE)
# -------------------------------

def test_handle_missing_values_datatype_imputation_mode1(sample_df1):
    df = sample_df1.copy()
    df.loc[0, "B"] = "x"
    result = handle_missing_values_datatype_imputation(df, NumericDatatypeImputationMethod.MODE)
    assert result["B"].isna().sum() == 0
    # "x" is the most appeared item in B column
    assert result.loc[3, "B"] == "x"


def test_handle_missing_values_datatype_imputation_mode2(sample_df2):
    df = sample_df2.copy()
    df.loc[2, "B"] = 4
    data = handle_missing_values_datatype_imputation(df, NumericDatatypeImputationMethod.MODE)

    # Mode of column "A" -> 1 is the mode (The first in the Mode list)
    assert data["A"][2] == 1  
    # After Modifications, Mode of column "B" -> 4 is the mode
    assert data["B"][0] == 4 
    # Mode of column "C" -> "a" is the mode
    assert data["C"][3] == "a"  

# -------------------------------
# Test handle_missing_values_adjacent_value_imputation (FORWARD)
# -------------------------------

def test_handle_missing_values_adjacent_value_imputation_forward1(sample_df1):
    df = sample_df1.copy()
    result = handle_missing_values_adjacent_value_imputation(df, AdjacentImputationMethod.FORWARD)
    assert result.isna().sum().sum() == 1  
    # B[0] whould not get affected
    result.loc[0, "B"] == None


def test_handle_missing_values_adjacent_value_imputation_forward2(sample_df2):
    data = handle_missing_values_adjacent_value_imputation(sample_df2, AdjacentImputationMethod.FORWARD)

    assert data["A"][0] == 1
    # Forward fill should replace None in "A" with 2
    assert data["A"][2] == 2  
    # Forward fill should replace None in "B" with 2
    assert np.isnan(data["B"][0])
    # Forward fill should replace None in "C" with "c"  
    assert data["C"][3] == "c"  

# -------------------------------
# Test handle_missing_values_adjacent_value_imputation (BACKWARD)
# -------------------------------

def test_handle_missing_values_adjacent_value_imputation_backward1(sample_df1):
    df = sample_df1.copy()
    result = handle_missing_values_adjacent_value_imputation(df, AdjacentImputationMethod.BACKWARD)
    assert result.isna().sum().sum() == 1  
    # B[3] whould not get affected
    assert result.loc[3, "B"] == None


def test_handle_missing_values_adjacent_value_imputation_backward2(sample_df2):
    data = handle_missing_values_adjacent_value_imputation(sample_df2, AdjacentImputationMethod.BACKWARD)

    # Backward fill should replace None in "A" with 4
    assert data["A"][2] == 4  
    # Backward fill should replace None in "B" with 3
    assert data["B"][0] == 2  
    # C[3] whould not get affected
    assert data["C"][3] == None

# -------------------------------
# Test handle_missing_values_adjacent_value_imputation (INTERPOLATION_LINEAR)
# -------------------------------

def test_handle_missing_values_adjacent_value_imputation_interpolation_linear1(sample_df1):
    df = sample_df1.copy()
    df.loc[3, "C"] = "2023-01-08"
    result = handle_missing_values_adjacent_value_imputation(df, AdjacentImputationMethod.INTERPOLATION_LINEAR)
    assert result["A"].isna().sum() == 0 
    # It only check the linear trend in the main column
    assert result.loc[2, "A"] == 3


def test_handle_missing_values_adjacent_value_imputation_interpolation_linear2(sample_df2):
    data = handle_missing_values_adjacent_value_imputation(sample_df2, AdjacentImputationMethod.INTERPOLATION_LINEAR)

    # Interpolation should replace None in "A" with 3
    assert data["A"][2] == pytest.approx(3.0, rel=1e-5)  
    # Interpolation don't replace, since there is no value behind
    assert np.isnan(data["B"][0]) 
    # "C" remains unchanged as it's categorical
    assert data["C"][3] == None 

# -------------------------------
# Test handle_missing_values_adjacent_value_imputation (INTERPOLATION_TIME)
# -------------------------------

def test_handle_missing_values_adjacent_value_imputation_interpolation_time1(sample_df1):
    df = sample_df1.copy()
    df.loc[3, "C"] = pd.to_datetime("2023-01-08")
    result = handle_missing_values_adjacent_value_imputation(df, AdjacentImputationMethod.INTERPOLATION_TIME, "C")
    assert result["A"].isna().sum() == 0 
    # It check the time-based trend of the datatime column
    assert result.loc[2, "A"] == pytest.approx(2.33333, rel=1e-5)


def test_handle_missing_values_adjacent_value_imputation_interpolation_time_invalid_timecolumn():
    df = pd.DataFrame({
        "date": ["a", "b", "c", "d"],
        "value": [1.0, None, 3.0, 4.0]
    })
    result = handle_missing_values_adjacent_value_imputation(df.copy(), AdjacentImputationMethod.INTERPOLATION_TIME, "date")
    assert result.empty