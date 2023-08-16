from pydantic import BaseModel
from enum import Enum

class Sex(Enum):
    male = "male"
    female = "female"

class Embarked(Enum):
    S = "S"
    C = "C"
    Q = "Q"

class PredictionRequest(BaseModel):
    PassengerId: float
    Pclass: int
    Name: str
    Sex: Sex
    Age: int
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: Embarked


class PredictionResponse(BaseModel):
    Survived: int
    