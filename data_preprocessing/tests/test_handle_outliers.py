import pytest
import pandas as pd
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handle_outliers import (
    detect_outliers, 
    handle_outliers,
    DetectOutlierMethod,
    HandleOutlierMethod,
)


@pytest.fixture
def sample_df():
    # This will create a DataFrame for testing purposes
    data = {
        "A" : [2, 3, 4, 5, 6, 8, 10, 11, 14, 15],
        "B" : [10, 12, 99, 100, 102, 103, 105, 107, 300, 500],
        "C" : [5, 7, 10, 11, 13, 100, 15, 18, 20, 17]
    }
    return pd.DataFrame(data)

# -------------------------------
# Test detect = IQR and all handle methods
# -------------------------------

def test_IQR_DROP(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.IQR)
    result = handle_outliers(data, HandleOutlierMethod.DROP, outliers)
    assert len(result) < len(data)
    # There are 4 outliers in B and 1 in C
    assert result["B"].to_list() == [99, 100, 102, 105, 107]


def test_IQR_REPLACE_WITH_MEDIAN(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.IQR)
    result = handle_outliers(data, HandleOutlierMethod.REPLACE_WITH_MEDIAN, outliers)
    # No row will be removed
    assert len(result) == len(sample_df)
    assert result.loc[0, "B"] == sample_df["B"].median()
    assert result.loc[5, "C"] == sample_df["C"].median()
    assert result["A"].equals(sample_df["A"])


def test_IQR_CAP_WITH_BOUNDRIES(sample_df):
    data = sample_df.copy()
    outliers, boundries = detect_outliers(data, DetectOutlierMethod.IQR)
    result = handle_outliers(data, HandleOutlierMethod.CAP_WITH_BOUNDARIES, outliers, boundries)
    # No row will be removed
    assert len(result) == len(sample_df)
    assert result.loc[0, "B"] == pytest.approx(88.357, rel=1e-3)
    assert result.loc[5, "C"] == 29.0
    assert result["A"].equals(sample_df["A"].astype("float"))


# -------------------------------
# Test detect = ZSCORE and all handle methods
# -------------------------------

def test_ZSCORE_DROP(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.ZSCORE)
    print(sample_df["B"].mean(), sample_df["B"].std())
    result = handle_outliers(data, HandleOutlierMethod.DROP, outliers)
    # No outlier is detected by ZSCORE method!
    assert len(result) == len(data)
    assert result.equals(data)


def test_ZSCORE_REPLACE_WITH_MEDIAN(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.ZSCORE)
    result = handle_outliers(data, HandleOutlierMethod.REPLACE_WITH_MEDIAN, outliers)
    # No outlier is detected by ZSCORE method!
    assert len(result) == len(data)
    assert result.equals(data)


def test_ZSCORE_CAP_WITH_BOUNDRIES(sample_df):
    data = sample_df.copy()
    outliers, boundries = detect_outliers(data, DetectOutlierMethod.ZSCORE)
    result = handle_outliers(data, HandleOutlierMethod.CAP_WITH_BOUNDARIES, outliers, boundries)
    # No outlier is detected by ZSCORE method!
    assert len(result) == len(data)
    assert result.equals(data)


# -------------------------------
# Test detect = ISOLATION_FOREST and all handle methods
# -------------------------------

def test_ISOLATION_FOREST_DROP(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.ISOLATION_FOREST)
    result = handle_outliers(data, HandleOutlierMethod.DROP, outliers)
    assert len(result) < len(data)
    # outliers: {'A': [0, 8, 9], 'B': [0, 1, 8, 9], 'C': [0, 5, 8]}
    assert result["B"].to_list() == [99, 100, 102, 105, 107]


def test_ISOLATION_FOREST_REPLACE_WITH_MEDIAN(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.ISOLATION_FOREST)
    result = handle_outliers(data, HandleOutlierMethod.REPLACE_WITH_MEDIAN, outliers)
    # No row will be removed
    assert len(result) == len(sample_df)
    assert result.loc[9, "A"] == sample_df["A"].median()
    assert result.loc[0, "B"] == sample_df["B"].median()
    assert result.loc[5, "C"] == sample_df["C"].median()


def test_ISOLATION_FOREST_CAP_WITH_BOUNDRIES(sample_df):
    data = sample_df.copy()
    outliers, boundries = detect_outliers(data, DetectOutlierMethod.ISOLATION_FOREST)
    result = handle_outliers(data, HandleOutlierMethod.CAP_WITH_BOUNDARIES, outliers, boundries)
    # No row will be removed
    assert len(result) == len(sample_df)
    assert result.loc[1, "A"] == 3.0
    assert result.loc[0, "B"] == 99.0
    assert result.loc[5, "C"] == 18.0


# -------------------------------
# Test detect = LOCAL_OUTLIER_FACTOR and all handle methods
# -------------------------------

def test_LOCAL_OUTLIER_FACTOR_DROP(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.LOCAL_OUTLIER_FACTOR, n_neighbors=3)
    result = handle_outliers(data, HandleOutlierMethod.DROP, outliers)
    assert len(result) < len(data)
    # outliers: {'A': [], 'B': [0, 1, 8, 9], 'C': [5]}
    assert result["B"].to_list() == [99, 100, 102, 105, 107]


def test_LOCAL_OUTLIER_FACTOR_REPLACE_WITH_MEDIAN(sample_df):
    data = sample_df.copy()
    outliers, _ = detect_outliers(data, DetectOutlierMethod.LOCAL_OUTLIER_FACTOR, n_neighbors=3)
    result = handle_outliers(data, HandleOutlierMethod.REPLACE_WITH_MEDIAN, outliers)
    # No row will be removed
    assert len(result) == len(sample_df)
    assert result.loc[0, "B"] == sample_df["B"].median()
    assert result.loc[5, "C"] == sample_df["C"].median()
    assert result["A"].equals(sample_df["A"])

def test_LOCAL_OUTLIER_FACTOR_CAP_WITH_BOUNDRIES(sample_df):
    data = sample_df.copy()
    outliers, boundries = detect_outliers(data, DetectOutlierMethod.LOCAL_OUTLIER_FACTOR, n_neighbors=3)
    result = handle_outliers(data, HandleOutlierMethod.CAP_WITH_BOUNDARIES, outliers, boundries)
    # No row will be removed
    assert len(result) == len(sample_df)
    assert result.loc[0, "B"] == 99.0
    assert result.loc[5, "C"] == 20.0
    assert result["A"].equals(sample_df["A"])