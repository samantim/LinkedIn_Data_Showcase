import pytest
import pandas as pd
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handle_duplicate_values import handle_duplicate_values_exact, handle_duplicate_values_fuzzy, config_logging

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'First Name': ['Alice', 'Alice', 'Bob', 'Charlie', 'Charlie', 'Charlie'],
        'Last Name': ['Smith', 'Smith', 'Jones', 'Brown', 'Brown', 'Browne'],
        'Age': [30, 30, 25, 40, 35, 40]
    })

@pytest.fixture
def fuzzy_data():
    return pd.DataFrame({
        'Name': ['Alice Smith', 'Alic Smith', 'Bob Jones', 'Charlie Brown', 'Charli Browne'],
        'City': ['Berlin', 'Berlin', 'Munich', 'Hamburg', 'Hamburg']
    })


# -------------------------------
# Test handle_duplicate_values_exact
# -------------------------------

def test_exact_duplicates_all_columns(sample_data):
    cleaned_df = handle_duplicate_values_exact(sample_data)
    # One exact duplicate should be dropped
    assert len(cleaned_df) == 5  
    assert cleaned_df.duplicated().sum() == 0

def test_exact_duplicates_subset(sample_data):
    cleaned_df = handle_duplicate_values_exact(sample_data, subset=['First Name', 'Last Name'])
    # Two duplicates (Charlie, Alice) on the subset
    assert len(cleaned_df) == 4  
    assert cleaned_df.duplicated(subset=['First Name', 'Last Name']).sum() == 0

def test_exact_duplicates_empty_df():
    empty_df = pd.DataFrame(columns=['A', 'B'])
    cleaned_df = handle_duplicate_values_exact(empty_df)
    # Check for not getting error in this case
    assert cleaned_df.empty 

# -------------------------------
# Test handle_duplicate_values_fuzzy
# -------------------------------

def test_fuzzy_duplicates_default_range(fuzzy_data):
    cleaned_df = handle_duplicate_values_fuzzy(fuzzy_data)
    # Expecting at least 'Alice Smith' and 'Alic Smith' to be grouped
    assert len(cleaned_df) < len(fuzzy_data)

def test_fuzzy_duplicates_custom_range(fuzzy_data):
    cleaned_df = handle_duplicate_values_fuzzy(fuzzy_data, ratio_range=(95, 100))
    # With stricter range, fewer fuzzy matches should be found
    assert len(cleaned_df) <= len(fuzzy_data)

def test_fuzzy_duplicates_on_subset(fuzzy_data):
    cleaned_df = handle_duplicate_values_fuzzy(fuzzy_data, subset=['City'], ratio_range=(95, 99))
    # No rows should be dropped, because all cities are exact matches
    assert len(cleaned_df) == len(fuzzy_data)  

def test_fuzzy_no_matches_with_high_threshold(fuzzy_data):
    cleaned_df = handle_duplicate_values_fuzzy(fuzzy_data, ratio_range=(100, 100))
    # With 100% threshold, behaves like exact match with no real duplicates
    assert cleaned_df.equals(fuzzy_data)  

def test_fuzzy_duplicates_empty_df():
    empty_df = pd.DataFrame(columns=['A', 'B'])
    cleaned_df = handle_duplicate_values_fuzzy(empty_df)
    # Check for not getting error in this case
    assert cleaned_df.empty 

