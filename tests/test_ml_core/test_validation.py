import pytest
from unittest.mock import MagicMock, patch
from src.ml_core.validation import validate  # Import the module containing the 'train' function
import src.utils.data_functions  as data_functions
from src.ml_pipelines.model_training import ModelTraining
import logging

LOGGER = logging.getLogger(__name__)

class TestModel:
    def score (self, x,y):
        return 1

def load(x):
    return x,x

def model(self):
    return TestModel

def report(mod, x,y,train):
    pass

#see if the train function calls all functions, each function is tested separately in other tests
def test_train_calls_load_data(monkeypatch,caplog): 
    monkeypatch.setattr("src.utils.data_functions.load_data", load)
    monkeypatch.setattr("src.utils.data_functions.load_model", model)
    monkeypatch.setattr("src.utils.data_functions.generate_validation_report", report)

    validate()