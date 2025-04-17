import pytest
import pandas as pd
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from convert_datatype import convert_datatype_auto, convert_datatype_ud


def test_convert_datatype_auto():
    df = pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Age": ["25", "30"],
        "Score": ["89.5", "92.3"],
        "Test Date": ["2024-04-10", "2024-04-12"]
    })
    converted_df = convert_datatype_auto(df)
    print(converted_df["Age"].dtype)
    assert pd.api.types.is_integer_dtype(converted_df["Age"]) or pd.api.types.is_float_dtype(converted_df["Age"])
    assert pd.api.types.is_float_dtype(converted_df["Score"])
    assert pd.api.types.is_datetime64_any_dtype(converted_df["Test Date"])


def test_convert_datatype_ud_success():
    df = pd.DataFrame({
        "High School Percentage": ["80", "90"],
        "Test Date": ["04/10/2024", "04/12/2024"]
    })
    scenario = {
        "column": ["High School Percentage", "Test Date"],
        "datatype": ["float", "datetime"],
        "format": ["", "%m/%d/%Y"]
    }
    converted_df = convert_datatype_ud(df, scenario)
    assert pd.api.types.is_float_dtype(converted_df["High School Percentage"])
    assert pd.api.types.is_datetime64_any_dtype(converted_df["Test Date"])


def test_convert_datatype_ud_fail_column():
    df = pd.DataFrame({
        "Some Column": ["80", "90"]
    })
    scenario = {
        "column": ["Not Exist"],
        "datatype": ["int"],
        "format": [""]
    }
    result_df = convert_datatype_ud(df, scenario)
    # No change should happen
    assert result_df.equals(df)  


def test_convert_datatype_ud_fail_type():
    df = pd.DataFrame({
        "Some Column": ["80", "90"]
    })
    scenario = {
        "column": ["Some Column"],
        "datatype": ["strange_type"],
        "format": [""]
    }
    result_df = convert_datatype_ud(df, scenario)
    assert result_df.equals(df)


def test_convert_datatype_ud_fail_defected_scenario():
    df = pd.DataFrame({
        "col1": ["80", "90"]
    })
    scenario = {
        "column": ["col1"],
        "datatype": ["dtype1", "dtype2"],
        "format": [""]
    }
    result_df = convert_datatype_ud(df, scenario)
    assert result_df.equals(df)
