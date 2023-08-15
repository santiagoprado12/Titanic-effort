# Data processing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from src.ml_pipelines.pipeline_connection import ModelTraining

from joblib import load
from io import BytesIO

import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)


def preprocess_data(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Preprocess the data"""

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
    Args:
        path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    df = pd.read_csv(path)
    X, y = preprocess_data(df, 'Survived')

    return X, y


def load_model(path: str) -> ModelTraining:
    """Load model from a pickle file.
    Args:
        path (str): Path to the pickle file.
    
    Returns:
        ModelTraining: Model.
    """
    with open(path, 'rb') as f:
        model = load(BytesIO(f.read()))
    return model


def main():

    logger.info('Loading data')
    X, y = load_data('data/validation.csv')

    logger.info('Loading model')
    model = load_model('models/best_model.pkl')

    logger.info('validating model')
    score = model.score(X, y)

    logger.info('The model has a score of %s on validation data', score)

