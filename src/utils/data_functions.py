# Data processing
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from joblib import load
from io import BytesIO



def preprocess_data(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Preprocess the data
    Its importat that the data has those columns:
        - Cabin
        - PassengerId
        - Name
        - Ticket
        - SibSp
        - Parch
    """

    titanic_data = data.copy()
    titanic_data.drop(['Cabin', 'PassengerId', 'Name',
                      'Ticket'], axis=1, inplace=True)
    titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']
    titanic_data['IsAlone'] = 0
    titanic_data.loc[titanic_data['FamilySize'] == 0, 'IsAlone'] = 1

    X = titanic_data.drop(target_column, axis=1)
    y = titanic_data[target_column]

    return X, y


def load_data(path: str) -> pd.DataFrame:   
    """Load data from a CSV file.
    It is important that the data has those columns:
        - Survived
    """

    df = pd.read_csv(path)
    X, y = preprocess_data(df, 'Survived')

    return X, y


def load_model(path: str) -> Pipeline:
    """Load model from a pickle file.
    Args:
        path (str): Path to the pickle file.
    
    Returns:
        Pipeline: The model.
    """
    with open(path, 'rb') as f:
        model = load(BytesIO(f.read()))
    return model
