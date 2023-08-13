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

dvc-pull-data:
	dvc pull -r data_remote --force

dvc-pull-model:
	dvc pull -r model_remote --force

train:
	make dvc-pull-data
	make dvc-pull-model
	python3 -m src.ml-core.train
	dvc unprotect 'models/best_model.pkl'
	dvc add 'models/best_model.pkl' --to-remote -r model_remote
	dvc push models/best_model.pkl.dvc -r model_remote

	