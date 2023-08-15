from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from src.ml_pipelines.pipeline_connection import *
from sklearn.dummy import DummyClassifier
import pytest
import os
from joblib import load
from io import BytesIO

def mock_init(self):
    # Mock the __init__() method to do nothing
    pass

def test_build_data_processing_pipeline_returns_col_transformer(monkeypatch):

    numeric_features = ['num1', 'num2']
    ordinal_attributes = ['ord1', 'ord2']
    categorical_features = ['cat1', 'cat2']

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    pipeline = PipelineBuilding()._build_data_processing_pipeline(
        numeric_features, ordinal_attributes, categorical_features
    )
    assert isinstance(pipeline, ColumnTransformer)


def test_build_data_processing_pipeline_process_all_columns_and_rows(monkeypatch):

    data = {
        'num1': [1, 2, 3, 4, 5],
        'ord1': [1, 0, 1, 0, 1],
        'cat1': ['a', 'b', 'a', 'b', 'a']
    }

    data = pd.DataFrame(data, columns=['num1', 'ord1', 'cat1'])

    numeric_features = ['num1']
    ordinal_attributes = ['ord1']
    categorical_features = ['cat1']

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    pipeline = PipelineBuilding()._build_data_processing_pipeline(
        numeric_features, ordinal_attributes, categorical_features
    )

    processed_data = pipeline.fit_transform(data)

    print(processed_data)

    assert processed_data.shape[1] == data.shape[1] + 1 # 1 bcecause cat1 has 2 values possible (one hot encoding)
    assert processed_data.shape[0] == 5


def test_build_processing_and_feature_selection_pipeline_returns_pipeline(monkeypatch):

    numeric_features = ['num1']
    ordinal_attributes = ['ord1']
    categorical_features = ['cat1', 'cat2']

    total_columns = numeric_features + ordinal_attributes + categorical_features

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    mock_pipeline = Pipeline([
        ('', SimpleImputer())
    ])

    pipeline = PipelineBuilding()._build_processing_and_feature_selection_pipeline(
        mock_pipeline, total_columns
    )

    assert isinstance(pipeline, Pipeline)



def test_train_models(monkeypatch):


    monkeypatch.setattr(ModelTraining, "__init__", mock_init)

    model_train = ModelTraining()

    model_train.X = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'ord1': [1, 3, 4, 5, 6],})
    
    model_train.y = pd.DataFrame({
        'target': [0,1,0,1,0]
    })

    model_train.models = {
        'model1': DummyClassifier()
    }

    models = model_train.train_models()

    try:
        models['model1'].predict(model_train.X)
    except Exception as e:
        pytest.fail(f"The model is not trained correctly: {e}")

    assert isinstance(models, dict)
    assert len(models) == 1
    assert isinstance(models['model1'], DummyClassifier) 



def test_generate_scores(monkeypatch):

    monkeypatch.setattr(ModelTraining, "__init__", mock_init)

    model_train = ModelTraining()

    model_train.X = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'ord1': [1, 3, 4, 5, 6],})
    
    model_train.y = pd.DataFrame({
        'target': [1, 1, 1, 1, 0]
    })

    model_train.models = {
        'model1': DummyClassifier(strategy="most_frequent")
    }

    model_train.train_models()

    scores = model_train.generate_scores(model_train.X, model_train.y)

    assert isinstance(scores, dict)
    assert len(scores) == 1
    assert isinstance(scores['model1'], float)
    assert scores['model1'] == 0.8


def test_best_model(monkeypatch):

    monkeypatch.setattr(ModelTraining, "__init__", mock_init)
    monkeypatch.setattr(ModelTraining, "generate_scores", lambda self, x, y: {'model1': 0.3, 'model2': 0.9})

    model_train = ModelTraining()
    best_model = model_train.best_model(1,2)

    assert isinstance(best_model, str)
    assert best_model == 'model2'


def test_save_model(monkeypatch):

    monkeypatch.setattr(ModelTraining, "__init__", mock_init)
    model = DummyClassifier().fit([1,2,3], [1,2,3])

    model_train = ModelTraining()
    model_train.models = { 
        'model1': model
    }

    model_train.save_model('model1', "model1.pkl")

    assert os.path.exists('model1.pkl')

    with open('model1.pkl', 'rb') as f:
        model_loaded = load(BytesIO(f.read()))
    
    assert isinstance(model_loaded, DummyClassifier)

    os.remove('model1.pkl')


    



