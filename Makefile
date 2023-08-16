VENV=venv
virtual_env:
	python3 -m pip install virtualenv
	python3 -m virtualenv $(VENV)

install: virtual_env
	. $(VENV)/bin/activate && python3 -m pip install --upgrade pip
	. $(VENV)/bin/activate && python3 -m pip install -r requirements.txt
	. $(VENV)/bin/activate && python3 -m pip install -r api/requirements.txt

test: 
	. $(VENV)/bin/activate && python3 -m coverage run --omit '/usr/*' -m pytest -v

test-coverage:
	. $(VENV)/bin/activate && python3 -m coverage run --omit '/usr/*' -m pytest -v
	. $(VENV)/bin/activate && python3 -m coverage report

coverage-html:
	. $(VENV)/bin/activate && python3 -m coverage html

coverage-report:
	. $(VENV)/bin/activate && python3 -m coverage report

dvc-pull-data:
	dvc pull data/train.csv -r data_remote
	dvc pull data/validation.csv -r data_remote

dvc-pull-model:
	dvc pull models/best_model.pkl -r model_remote

dvc-push-train-data:
	dvc unprotect 'data/train.csv'
	dvc add 'data/train.csv' --to-remote -r data_remote
	dvc push data/train.csv.dvc -r data_remote

dvc-push-validation-data:
	dvc unprotect 'data/validation.csv'
	dvc add 'data/validation.csv' --to-remote -r data_remote
	dvc push data/validation.csv.dvc -r data_remote

dvc-push-data:
	dvc unprotect 'data/train.csv'
	dvc unprotect 'data/validation.csv'
	dvc add 'data/train.csv' --to-remote -r data_remote
	dvc add 'data/validation.csv' --to-remote -r data_remote
	dvc push data/train.csv.dvc -r data_remote
	dvc push data/validation.csv.dvc -r data_remote

train:
	make dvc-pull
	python3 -m src.CLI train --model=knn --model=random_forest --model=gradient_boosting -th=0.7 -rm
	
train-git-actions:
	python3 -m src.CLI train --model=knn --model=random_forest --model=gradient_boosting -th=0.7 -rm --git-actions

upload_new_dataset:
	dvc unprotect 'data/validation.csv'
	dvc unprotect 'data/train.csv'
	dvc add 'data/validation.csv' --to-remote -r data_remote
	dvc add 'data/train.csv' --to-remote -r data_remote

register_model:
	dvc unprotect 'models/best_model.pkl'
	dvc add 'models/best_model.pkl' --to-remote -r model_remote
	dvc push models/best_model.pkl.dvc -r model_remote

upload_new_dataset-creds: 
	python -m src.test

create-docker-image:
	aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 019994626350.dkr.ecr.us-east-2.amazonaws.com
	docker build -t titanic-api .
	docker tag titanic-api:latest 019994626350.dkr.ecr.us-east-2.amazonaws.com/titanic-api:latest
	docker push 019994626350.dkr.ecr.us-east-2.amazonaws.com/titanic-api:latest


dummy:
	@echo "Doing nothing"

