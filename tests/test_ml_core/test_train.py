from src.ml_core.train import *
from unittest.mock import patch
import pandas as pd

@patch('pandas.read_csv')
def test_load_data(mock_read_csv):
    # Define the mocked return value for pd.read_csv
    mock_read_csv.return_value = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    })

    # Call the function with a dummy path
    path = 'dummy.csv'
    result = load_data(path)

    # Assert that pd.read_csv was called with the expected path
    mock_read_csv.assert_called_once_with(path)

    # Assert that the result is a DataFrame with the mocked data
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.columns) == ['column1', 'column2']


def test_preprocess_data():
    # Define the mocked input DataFrame
    input_data = pd.DataFrame({
        'PassengerId': [1, 2, 3],
        'Name': ['John', 'Jane', 'Alice'],
        'Cabin': ['A123', 'B456', 'C789'],
        'Ticket': ['T123', 'T456', 'T789'],
        'SibSp': [0, 1, 2],
        'Parch': [0, 0, 1]
    })

    # Define the expected output DataFrame after preprocessing
    expected_output = pd.DataFrame({
        'SibSp': [0, 1, 2],
        'Parch': [0, 0, 1],
        'FamilySize': [0, 1, 3],
        'IsAlone': [1, 0, 0]
    })

    # Call the function with the mocked input data
    result = preprocess_data(input_data)
    
    print(result)
    # Assert that the result matches the expected output
    pd.testing.assert_frame_equal(result, expected_output)