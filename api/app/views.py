from .models import PredictionRequest
from .utils import get_model, transform_to_dataframe
from src.utils.data_functions import preprocess_features

model = get_model()

def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    data_to_predict = preprocess_features(data_to_predict)
    prediction = model.predict(data_to_predict)[0]
    return max(prediction, 0)