import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        'A': [1, 2, None, 4],
        'B': [None, 2, 3, 4],
        'C': ['a', 'a', 'c', None]
    }
    return pd.DataFrame(data)


def test_handle_missing_values_drop(sample_df1):
    result = handle_missing_values_drop(sample_df1.copy())
    assert result.isna().sum().sum() == 0
    assert len(result) == 1  

def test_impute_mean(sample_df1):
    df = sample_df1.copy()
    df.loc[0, 'A'] = None
    result = handle_missing_values_datatype_imputation(df, NumericDatatypeImputationMethod.MEAN)
    assert result['A'].isna().sum() == 0

def test_impute_median(sample_df1):
    df = sample_df1.copy()
    df.loc[0, 'A'] = None
    result = handle_missing_values_datatype_imputation(df, NumericDatatypeImputationMethod.MEDIAN)
    assert result['A'].isna().sum() == 0

def test_impute_mode(sample_df1):
    df = sample_df1.copy()
    df.loc[1, 'B'] = None
    result = handle_missing_values_datatype_imputation(df, NumericDatatypeImputationMethod.MODE)
    assert result['B'].isna().sum() == 0

def test_forward_fill(sample_df1):
    df = sample_df1.copy()
    result = handle_missing_values_adjacent_value_imputation(df, AdjacentImputationMethod.Forward)
    assert result.isna().sum().sum() == 1  

def test_backward_fill(sample_df1):
    df = sample_df1.copy()
    result = handle_missing_values_adjacent_value_imputation(df, AdjacentImputationMethod.Backward)
    assert result.isna().sum().sum() == 1  

def test_linear_interpolation(sample_df1):
    df = sample_df1.copy()
    result = handle_missing_values_adjacent_value_imputation(df, AdjacentImputationMethod.Interpolation_Linear)
    assert result['A'].isna().sum() == 0 


# Test for handle_missing_values_drop function
def test_handle_missing_values_drop(sample_df2):
    data = handle_missing_values_drop(sample_df2)

    # Check that rows with missing values are dropped
    assert data.shape[0] == 1  # After dropping rows with NaN, only one row should remain
    assert 'A' in data.columns
    assert 'B' in data.columns
    assert 'C' in data.columns


# Test for handle_missing_values_datatype_imputation function with MEAN
def test_handle_missing_values_datatype_imputation_mean(sample_df2):
    data = handle_missing_values_datatype_imputation(sample_df2, NumericDatatypeImputationMethod.MEAN)

    # Test that NaN values in numeric columns are replaced by the mean of that column
    assert data['A'][2] == pytest.approx(2.333333, rel=1e-5)  # Mean of column 'A' -> (1 + 2 + 4) / 3 = 2.3333...
    assert data['B'][0] == pytest.approx(3.0, rel=1e-5)  # Mean of column 'B' -> (2 + 3 + 4) / 3 = 3.0
    assert data['C'][3] == 'a'  # 'C' is a categorical column, no change in NaN for 'C'


# Test for handle_missing_values_datatype_imputation function with MEDIAN
def test_handle_missing_values_datatype_imputation_median(sample_df2):
    data = handle_missing_values_datatype_imputation(sample_df2, NumericDatatypeImputationMethod.MEDIAN)

    # Test that NaN values in numeric columns are replaced by the median of that column
    assert data['A'][2] == 2.0  # Median of column 'A' -> (2 is the median)
    assert data['B'][0] == 3.0  # Median of column 'B' -> (3 is the median)
    assert data['C'][3] == 'a'  # 'C' is a categorical column, no change in NaN for 'C'


# Test for handle_missing_values_datatype_imputation function with MODE
def test_handle_missing_values_datatype_imputation_mode(sample_df2):
    data = handle_missing_values_datatype_imputation(sample_df2, NumericDatatypeImputationMethod.MODE)

    # Test that NaN values in numeric columns are replaced by the mode of that column
    assert data['A'][2] == 1  # Mode of column 'A' -> 1 is the mode
    assert data['B'][0] == 2  # Mode of column 'B' -> 2 is the mode
    assert data['C'][3] == 'a'  # Mode of column 'C' -> 'a' is the mode


# Test for handle_missing_values_adjacent_value_imputation with Forward fill
def test_handle_missing_values_adjacent_value_imputation_forward(sample_df2):
    data = handle_missing_values_adjacent_value_imputation(sample_df2, AdjacentImputationMethod.Forward)

    # Test that NaN values are replaced with the next value in the column (forward fill)
    assert data['A'][0] == 1
    assert data['A'][2] == 2  # Forward fill should replace None in 'A' with 2
    assert np.isnan(data['B'][0])  # Forward fill should replace None in 'B' with 2
    assert data['C'][3] == 'c'  # Forward fill should replace None in 'C' with 'c'


# Test for handle_missing_values_adjacent_value_imputation with Backward fill
def test_handle_missing_values_adjacent_value_imputation_backward(sample_df2):
    data = handle_missing_values_adjacent_value_imputation(sample_df2, AdjacentImputationMethod.Backward)

    # Test that NaN values are replaced with the previous value in the column (backward fill)
    assert data['A'][2] == 4  # Backward fill should replace None in 'A' with 4
    assert data['B'][0] == 2  # Backward fill should replace None in 'B' with 3
    assert data['C'][3] == None  # 'C' should be replaced with 'c'


# Test for handle_missing_values_adjacent_value_imputation with Linear Interpolation
def test_handle_missing_values_adjacent_value_imputation_interpolation_linear(sample_df2):
    data = handle_missing_values_adjacent_value_imputation(sample_df2, AdjacentImputationMethod.Interpolation_Linear)

    # Test that NaN values are interpolated for numeric columns (linear interpolation)
    assert data['A'][2] == pytest.approx(3.0, rel=1e-5)  # Interpolation should replace None in 'A' with 3
    assert np.isnan(data['B'][0]) # Interpolation should replace None in 'B' with 1
    assert data['C'][3] == None # 'C' remains unchanged as it's categorical


def test_time_interpolation():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4),
        "value": [1.0, None, None, 4.0]
    })
    result = handle_missing_values_adjacent_value_imputation(df.copy(), AdjacentImputationMethod.Interpolation_Time, "date")
    assert result["value"].isna().sum() == 0

def test_time_interpolation_invalid_time_column():
    df = pd.DataFrame({
        "date": ["a", "b", "c", "d"],
        "value": [1.0, None, 3.0, 4.0]
    })
    result = handle_missing_values_adjacent_value_imputation(df.copy(), AdjacentImputationMethod.Interpolation_Time, "date")
    assert result.empty