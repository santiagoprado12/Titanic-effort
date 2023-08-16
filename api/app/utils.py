from joblib import load
from scipy.sparse import data
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os 
from io import BytesIO

def get_model() -> Pipeline:
    model_path = os.environ.get('MODEL_PATH','models/best_model.pkl')
    with open(model_path,'rb') as model_file:
        model = load(BytesIO(model_file.read()))
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dictionary = {}
    for key, value in class_model.dict().items():
        if key in ('Sex','Embarked'):
            transition_dictionary[key] = [value.value]
        else:
            transition_dictionary[key] = [value]

    data_frame = DataFrame(transition_dictionary)
    return data_frame