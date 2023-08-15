VENV=venv
virtual_env:
	python3 -m pip install virtualenv
	python3 -m virtualenv $(VENV)

install: 
	pwd
	. $(VENV)/bin/activate && python3 -m pip install --upgrade pip
	. $(VENV)/bin/activate && python3 -m pip install -r requirements.txt

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
	dvc pull -r data_remote --force

dvc-pull-model:
	dvc pull -r model_remote --force

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

dummy:
	@echo "Doing nothing"