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

coverage-html: test-coverage
	. $(VENV)/bin/activate && python3 -m coverage html

coverage-report: test-coverage
	. $(VENV)/bin/activate && python3 -m coverage report