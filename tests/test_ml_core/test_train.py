import pytest
from unittest.mock import MagicMock, patch
from src.ml_core.train import train  # Import the module containing the 'train' function
import src.utils.data_functions  as data_functions
from src.ml_pipelines.model_training import ModelTraining
import logging

LOGGER = logging.getLogger(__name__)

def load(x):
    return x,x

def train_test_split(x, y, test, rend):
    return x,y,x,y

def __init__(self, x, y, x_test, y_test):
    pass

def train(self):
    pass

def save(self):
    pass

def report(mod, x,y,train):
    pass

#see if the train function calls all functions, each function is tested separately in other tests
def test_train_calls_load_data(monkeypatch,caplog): 
    monkeypatch.setattr("src.utils.data_functions.load_data", load)
    monkeypatch.setattr("sklearn.model_selection.train_test_split", train_test_split)
    monkeypatch.setattr("src.ml_pipelines.model_training.ModelTraining.__init__", __init__)
    monkeypatch.setattr("src.ml_pipelines.model_training.ModelTraining.train_models", train)
    monkeypatch.setattr("src.ml_pipelines.model_training.ModelTraining.save_model", save)
    monkeypatch.setattr("src.utils.data_functions.generate_validation_report", report)

    train([])