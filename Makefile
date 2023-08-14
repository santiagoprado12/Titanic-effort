VENV=venv
virtual_env:
	python3 -m pip install virtualenv
	python3 -m virtualenv $(VENV)

install: virtual_env
	. $(VENV)/bin/activate && python3 -m pip install --upgrade pip
	. $(VENV)/bin/activate && python3 -m pip install -r requirements.txt

test: 
	. $(VENV)/bin/activate && python3 -m pytest -v

test-coverage:
	. $(VENV)/bin/activate && python3 -m coverage run -m pytest -v
	. $(VENV)/bin/activate && python3 -m coverage report

coverage-html: test
	. $(VENV)/bin/activate && python3 -m coverage html

coverage-report: test
	. $(VENV)/bin/activate && python3 -m coverage report

dvc-pull:
	dvc pull

train:
	make dvc-pull
	python3 -m src.ml_core.train

train-creds:
	python -m src.test

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