import pytest
import pandas as pd
import numpy as np
import os 
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scale_feature import get_observing_columns, scale_feature


@pytest.fixture
def sample_df():
    data = {
        "Age": [18, 25, 40, 60],
        "High School Percentage": [70, 80, 90, 85],
        "City": ["Berlin", "Paris", "Madrid", "Rome"]
    }
    return pd.DataFrame(data)


# ---------------------
# Tests for get_observing_columns
# ---------------------

def test_get_valid_subset_numeric_columns(sample_df):
    result = get_observing_columns(sample_df, ["Age"])
    assert result == ["Age"]

def test_get_subset_with_non_numeric(sample_df):
    result = get_observing_columns(sample_df, ["Age", "City"])
    assert result == []

def test_get_subset_with_invalid_column(sample_df):
    result = get_observing_columns(sample_df, ["Age", "InvalidColumn"])
    assert result == []

def test_strip_whitespace_in_column_names(sample_df):
    result = get_observing_columns(sample_df, ["  Age  "])
    assert result == ["Age"]


# ---------------------
# Tests for scale_feature
# ---------------------

def test_minmax_scaling(sample_df):
    scale_config = {
        "column": ["Age"],
        "scaling_method": ["MINMAX_SCALING"]
    }
    result = scale_feature(sample_df.copy(), scale_config)
    expected = MinMaxScaler().fit_transform(sample_df[["Age"]])
    assert pytest.approx(result["Age"].tolist(), rel=1e-2) == expected.flatten().tolist()

def test_zscore_standardization(sample_df):
    scale_config = {
        "column": ["Age"],
        "scaling_method": ["ZSCORE_STANDARDIZATION"]
    }
    result = scale_feature(sample_df.copy(), scale_config)
    expected = StandardScaler().fit_transform(sample_df[["Age"]])
    assert pytest.approx(result["Age"].tolist(), rel=1e-2) == expected.flatten().tolist()

def test_robust_scaling(sample_df):
    scale_config = {
        "column": ["Age"],
        "scaling_method": ["ROBUST_SCALING"]
    }
    result = scale_feature(sample_df.copy(), scale_config)
    expected = RobustScaler().fit_transform(sample_df[["Age"]])
    assert pytest.approx(result["Age"].tolist(), rel=1e-2) == expected.flatten().tolist()

def test_l2_normalization(sample_df):
    scale_config = {
        "column": ["Age", "High School Percentage"],
        "scaling_method": ["MINMAX_SCALING", "MINMAX_SCALING"]
    }
    result = scale_feature(sample_df.copy(), scale_config, apply_l2normalization=True)
    
    # Select numeric columns from the result
    numeric_data = np.array(result.select_dtypes("number").values.tolist())
    print(numeric_data)
    
    # Calculate L2 norm (Euclidean norm) for each row
    l2_norms = (numeric_data**2).sum(axis=1)**0.5
    print(l2_norms)

    # Assert each row's L2 norm is approximately 1 unless the row cells are 0 which results in zero norm
    assert all(abs(norm - 1) < 1e-6 or norm == 0 for norm in l2_norms)

def test_invalid_scaling_method(sample_df):
    scale_config = {
        "column": ["Age"],
        "scaling_method": ["INVALID_METHOD"]
    }
    result = scale_feature(sample_df.copy(), scale_config)
    # Should return original unmodified data
    assert result.equals(sample_df)

def test_mismatched_columns_and_methods(sample_df):
    scale_config = {
        "column": ["Age", "High School Percentage"],
        "scaling_method": ["ZSCORE_STANDARDIZATION"]  # Only one method for two columns
    }
    result = scale_feature(sample_df.copy(), scale_config)
    # Should return original unmodified data
    assert result.equals(sample_df)
