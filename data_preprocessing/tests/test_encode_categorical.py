import pandas as pd
import pytest
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encode_categorical import get_observing_columns, encode_categorical, CategoricalEncodingMethod

# Sample DataFrame for testing
@pytest.fixture
def sample_data():
    return pd.DataFrame({
    "Color": ["Red", "Blue", "Green", "Red", "Blue"],
    "Shape": ["Circle", "Square", "Triangle", "Circle", "Square"],
    "Size": [1, 2, 3, 2, 1]  # Numeric column (should be excluded)
})


# -----------------------------
# Tests for get_observing_columns
# -----------------------------

def test_get_observing_columns_all_categorical(sample_data):
    result = get_observing_columns(sample_data, None)
    assert set(result) == {"Color", "Shape"}

def test_get_observing_columns_valid_subset(sample_data):
    result = get_observing_columns(sample_data, ["Color"])
    assert result == ["Color"]

def test_get_observing_columns_invalid_subset(sample_data):
    result = get_observing_columns(sample_data, ["Size"])
    assert result == []

def test_get_observing_columns_mixed_subset(sample_data):
    result = get_observing_columns(sample_data, ["Color", "Size"])
    assert result == []

# -----------------------------
# Tests for encode_categorical
# -----------------------------

def test_label_encoding_single_column(sample_data):
    df = sample_data.copy()
    encoded_df = encode_categorical(df, CategoricalEncodingMethod.LABEL_ENCODING, ["Color"])
    assert "Color_encoded" in encoded_df.columns
    assert encoded_df["Color_encoded"].dtype in ["int32", "int64"]

def test_label_encoding_all_columns(sample_data):
    df = sample_data.copy()
    encoded_df = encode_categorical(df, CategoricalEncodingMethod.LABEL_ENCODING)
    assert "Color_encoded" in encoded_df.columns
    assert "Shape_encoded" in encoded_df.columns

def test_onehot_encoding_single_column(sample_data):
    df = sample_data.copy()
    encoded_df = encode_categorical(df, CategoricalEncodingMethod.ONEHOT_ENCODING, ["Color"])
    assert any("Color_" in col for col in encoded_df.columns)

def test_onehot_encoding_all_columns(sample_data):
    df = sample_data.copy()
    encoded_df = encode_categorical(df, CategoricalEncodingMethod.ONEHOT_ENCODING)
    assert any("Color_" in col for col in encoded_df.columns)
    assert any("Shape_" in col for col in encoded_df.columns)

# caplog is a builtin pytest utility that captures log messages (produced by Python’s logging module) during test execution.
def test_hashing_encoding_warning(sample_data, caplog):
    df = sample_data.copy()
    encoded_df = encode_categorical(df, CategoricalEncodingMethod.HASHING, ["Color"])
    assert any(col.startswith("Color_") for col in encoded_df.columns)
    assert "Hashing for category number less than 10" in caplog.text

def test_hashing_encoding_all_columns(sample_data):
    df = sample_data.copy()
    encoded_df = encode_categorical(df, CategoricalEncodingMethod.HASHING)
    assert any("Color_" in col for col in encoded_df.columns)
    assert any("Shape_" in col for col in encoded_df.columns)

def test_invalid_column_subset_empty_return(sample_data):
    df = sample_data.copy()
    encoded_df = encode_categorical(df, CategoricalEncodingMethod.LABEL_ENCODING, ["Size"])
    # It should return the original df (no changes)
    assert df.equals(encoded_df)

