FROM python:3.10-slim-buster

WORKDIR /api

COPY api/requirements.txt .

COPY src/ ./src

RUN apt-get update && apt-get install -y make

RUN pip install -U pip && pip install -r requirements.txt

COPY api/ ./api

COPY Makefile .

COPY models/best_model.pkl ./models/best_model.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]