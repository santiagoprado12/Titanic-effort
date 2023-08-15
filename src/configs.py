# Data processing
import numpy as np

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV


class Configs:

    models = [
        {
            'name': 'random_forest',
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False],
                'random_state': [42]
            }
        },
        {
            'name': 'gradient_boosting',
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'subsample': [0.8, 0.9, 1.0],
                'random_state': [42]
            }
        },
        {
            'name': 'knn',
            'model': KNeighborsClassifier(),
            'params': {
            'n_neighbors': np.arange(1, 21),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        }

    ]
