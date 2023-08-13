# Data processing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from src.pipelines.model_trainig import ModelTraining

import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)



def load_data(path: str) -> pd.DataFrame:
    """Load data from a given path"""
    return pd.read_csv(path)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data"""

    titanic_data = data.copy()
    titanic_data.drop(['Cabin', 'PassengerId', 'Name',
                      'Ticket'], axis=1, inplace=True)
    titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']
    titanic_data['IsAlone'] = 0
    titanic_data.loc[titanic_data['FamilySize'] == 0, 'IsAlone'] = 1

    return titanic_data


if __name__ == "__main__":

    RANDOM_SEED = 42

    atributes_types = {
        'target': 'Survived',
        'ordinal_attributes': ['Sex', 'IsAlone'],
        'numeric_features': ['Parch', 'Pclass', 'SibSp', 'Fare', 'Age'],
        'categorical_features': ['Embarked', 'FamilySize']
    }

    models_to_use = ["random_forest", "gradient_boosting", "knn"]

    logger.info("Loading data")
    
    titanic_data = load_data("data/train.csv")
    titanic_data = preprocess_data(titanic_data)

    X = titanic_data.drop(atributes_types["target"], axis=1)
    y = titanic_data[atributes_types["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_SEED)
    
    logger.info("Training models")
    
    model_train = ModelTraining(X_train, y_train, atributes_types, models_to_use)
    models_trained = model_train.train_models()

    scores = model_train.generate_scores(X_test, y_test)

    logger.info(f"Scores: {scores}")
    best_model_name = model_train.best_model(X_test, y_test)

    logger.info(f"Best model: {best_model_name}")
    best_model = models_trained[best_model_name]

    assert scores[best_model_name] > 0.7, "The best model is not good enough, try with different hyperparams"

    logger.info("Saving model")
    model_train.save_model(best_model_name, "models/best_model.pkl")

