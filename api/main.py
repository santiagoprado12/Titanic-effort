from fastapi import FastAPI
from .app.models import PredictionRequest, PredictionResponse
from .app.views import get_prediction

app = FastAPI(docs_url="/",
              title="Titanic Effort",
            description="Inference endpoint for model trained on Titanic dataset.",
            version="1.0.0")

@app.post("/v1/prediction")
def make_model_prediction(request: PredictionRequest):
    return PredictionResponse(Survived=get_prediction(request))